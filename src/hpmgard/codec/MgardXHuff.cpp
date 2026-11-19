// Dispatcher of the MGARD-X Huffman stage: zigzag mapping, dictionary guard, device selection.
#include "hpmgard/codec/MgardXHuff.h"
#include <algorithm>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

#ifdef HPMGARD_HAVE_MGARDX
bool EncodeSerial(const std::vector<int32_t>& codes, int zstd_level, std::vector<uint8_t>& out);
bool DecodeSerial(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level);
#endif
#ifdef HPMGARD_HAVE_HIP
bool HipDevicePresent();
bool EncodeHip(const std::vector<int32_t>& codes, int zstd_level, std::vector<uint8_t>& out);
bool DecodeHip(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level);
#endif

namespace {
inline uint32_t Zigzag(int32_t v) { return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31); }
inline int32_t Unzigzag(uint32_t z) { return static_cast<int32_t>((z >> 1) ^ (0u - (z & 1u))); }
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

// MGARD-X serialises a ~34 KB codebook/metadata block per stream (dictionary 8192), so short streams are
// coded with the in-house canonical Huffman instead (the caller falls back when this returns false).
constexpr size_t kMinSymbols = size_t(1) << 16;

bool Encode(int device, const int32_t* sym, size_t n, int zstd_level, std::vector<uint8_t>& out) {
    if (!Available(device) || n < kMinSymbols) return false;
    std::vector<int32_t> codes(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t z = Zigzag(sym[i]);
        if (z >= kDictSize) return false;          // outside the MGARD-X dictionary: let the caller fall back
        codes[i] = static_cast<int32_t>(z);
    }
    if (n == 0) { out.clear(); return true; }
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1) return EncodeHip(codes, zstd_level, out);
#endif
#ifdef HPMGARD_HAVE_MGARDX
    return EncodeSerial(codes, zstd_level, out);
#else
    return false;
#endif
}

bool Decode(int device, const uint8_t* b, size_t nb, int32_t* out, size_t n, int zstd_level) {
    if (n == 0) return true;
    std::vector<int32_t> codes;
    bool ok = false;
#if defined(HPMGARD_HAVE_MGARDX) && defined(HPMGARD_HAVE_HIP)
    if (device == 1 && Available(1)) ok = DecodeHip(b, nb, codes, n, zstd_level);
#endif
#ifdef HPMGARD_HAVE_MGARDX
    if (!ok) ok = DecodeSerial(b, nb, codes, n, zstd_level);   // any backend decodes any stream
#endif
    if (!ok || codes.size() != n) return false;
    for (size_t i = 0; i < n; ++i) out[i] = Unzigzag(static_cast<uint32_t>(codes[i]));
    return true;
}

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
