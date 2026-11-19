// Dispatcher of the MGARD-X lossless stage: zigzag mapping (host path), dictionary guard, device selection,
// payload header (dict, block), warm-up bookkeeping.
#include "hpmgard/codec/MgardXHuff.h"
#include <algorithm>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

#ifdef HPMGARD_HAVE_MGARDX
bool EncodeSerial(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out);
bool DecodeSerial(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block);
#endif
#ifdef HPMGARD_HAVE_HIP
bool HipDevicePresent();
bool EncodeHip(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out);
bool EncodeHipDevice(const int32_t* d_sym, size_t n, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out);
bool DecodeHip(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block);
void WarmupHip(XType t, int level);
#endif

const char* XTypeName(XType t) {
    switch (t) {
    case XType::Huffman: return "Huffman";
    case XType::HuffmanZstd: return "Huffman_Zstd";
    case XType::ZeroRleRans: return "ZeroRLE_Rans";
    case XType::SymbolRans: return "SymbolRans";
    }
    return "?";
}

namespace {
inline uint32_t Zigzag(int32_t v) { return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31); }
inline int32_t Unzigzag(uint32_t z) { return static_cast<int32_t>((z >> 1) ^ (0u - (z & 1u))); }
void PutU32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); }
uint32_t GetU32(const uint8_t* p) { uint32_t x = 0; for (int i = 0; i < 4; ++i) x |= static_cast<uint32_t>(p[i]) << (8 * i); return x; }
void Finish(std::vector<uint8_t>& out, uint32_t dict, uint32_t block, const std::vector<uint8_t>& x) {
    out.clear(); out.reserve(x.size() + 8);
    PutU32(out, dict); PutU32(out, block);
    out.insert(out.end(), x.begin(), x.end());
}
}

// MGARD-X's rANS coders segfault on a one-symbol alphabet (SERIAL SymbolRans, all-identical input), so
// those types need at least two distinct codes; Huffman handles the degenerate case itself.
bool AlphabetOk(XType t, uint32_t zmin, uint32_t zmax) {
    if (t == XType::SymbolRans || t == XType::ZeroRleRans) return zmin != zmax;
    return true;
}

bool Available(int device) {
#ifdef HPMGARD_HAVE_MGARDX
    if (device == 0) return true;
#endif
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1) return HipDevicePresent();
#endif
    return false;
}

bool Encode(int device, const int32_t* sym, size_t n, XType t, int level, std::vector<uint8_t>& out) {
    if (!Available(device) || n < kMinSymbols) return false;
    const uint32_t dict = DictSize(), block = BlockSize();
    std::vector<int32_t> codes(n);
    uint32_t zmin = 0xFFFFFFFFu, zmax = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t z = Zigzag(sym[i]);
        if (z >= dict) return false;               // outside the MGARD-X dictionary: let the caller fall back
        codes[i] = static_cast<int32_t>(z);
        zmin = std::min(zmin, z); zmax = std::max(zmax, z);
    }
    if (!AlphabetOk(t, zmin, zmax)) return false;
    std::vector<uint8_t> x;
    bool ok = false;
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1) ok = EncodeHip(codes, t, level, dict, block, x);
    else
#endif
#ifdef HPMGARD_HAVE_MGARDX
    ok = EncodeSerial(codes, t, level, dict, block, x);
#else
    ok = false;
#endif
    if (!ok) return false;
    Finish(out, dict, block, x);
    return true;
}

bool EncodeDevice(int device, const int32_t* d_sym, size_t n, XType t, int level, std::vector<uint8_t>& out) {
    if (!d_sym || n < kMinSymbols) return false;
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1 && Available(1)) {
        std::vector<uint8_t> x;
        if (!EncodeHipDevice(d_sym, n, t, level, DictSize(), BlockSize(), x)) return false;
        Finish(out, DictSize(), BlockSize(), x);
        return true;
    }
#endif
    (void)device; (void)t; (void)level; (void)out;
    return false;
}

bool Decode(int device, const uint8_t* b, size_t nb, int32_t* out, size_t n, XType t, int level) {
    if (n == 0) return true;
    if (nb < 8) return false;
    const uint32_t dict = GetU32(b), block = GetU32(b + 4);
    b += 8; nb -= 8;
    std::vector<int32_t> codes;
    bool ok = false;
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1 && Available(1)) ok = DecodeHip(b, nb, codes, n, t, level, dict, block);
#endif
#ifdef HPMGARD_HAVE_MGARDX
    if (!ok) ok = DecodeSerial(b, nb, codes, n, t, level, dict, block);   // any backend decodes any stream
#endif
    (void)device;
    if (!ok || codes.size() != n) return false;
    for (size_t i = 0; i < n; ++i) out[i] = Unzigzag(static_cast<uint32_t>(codes[i]));
    return true;
}

void Warmup(int device, XType t) {
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device != 1 || !Available(1)) return;
    static std::mutex m; static std::set<int> done;
    std::lock_guard<std::mutex> lk(m);
    if (!done.insert(static_cast<int>(t)).second) return;
    WarmupHip(t, 3);
#else
    (void)device; (void)t;
#endif
}

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
