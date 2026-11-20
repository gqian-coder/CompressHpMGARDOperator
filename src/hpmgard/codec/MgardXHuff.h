// MGARD-X lossless coders as first-stage symbol coders: Huffman (optionally + zstd inside MGARD-X),
// ZeroRLE_Rans and SymbolRans, on the SERIAL (CPU) or HIP device. Streams are device independent, so a
// GPU-coded stream decodes on the CPU and vice versa. The device path (EncodeDevice) takes the quantised
// symbols from device memory: zigzag mapping runs on the device and nothing is copied to the host except
// the compressed bytes. Compressor objects (MGARD-X workspaces) persist per process and are reused.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

enum class XType : uint8_t { Huffman = 0, HuffmanZstd = 1, ZeroRleRans = 2, SymbolRans = 3 };
const char* XTypeName(XType t);

// Largest MGARD-X dictionary (zigzag codes >= it are outliers, see OutlierRatio) and the Huffman block size.
// Defaults 8192 / 4096; override with HPMGARD_XHUFF_DICT / HPMGARD_XHUFF_BLOCK (read once). Both are recorded
// in the stream. A larger dictionary makes MGARD-X's codebook construction much slower (65536: 2-3x slower
// coding on the VKI velocity), a smaller block raises the parallelism of its decoder (p4 pressure, 33M
// symbols: block 20480 decodes in 0.18 s, 4096 in 0.075 s, 1024 in 0.055 s, for +9% / +20% bytes).
inline uint32_t DictSize() {
    static const uint32_t v = [] { const char* e = std::getenv("HPMGARD_XHUFF_DICT"); const long x = e ? std::atol(e) : 0; return x >= 2 && x <= 1048576 ? static_cast<uint32_t>(x) : 8192u; }();
    return v;
}
inline uint32_t BlockSize() {
    static const uint32_t v = [] { const char* e = std::getenv("HPMGARD_XHUFF_BLOCK"); const long x = e ? std::atol(e) : 0; return x >= 64 && x <= (1l << 24) ? static_cast<uint32_t>(x) : 4096u; }();
    return v;
}
// Codes >= the dictionary are stored by MGARD-X as outliers (index + value) when they are at most this
// fraction of the stream (HPMGARD_XHUFF_OUTLIER, default 0.01); above it the stream falls back to the
// in-house Huffman. The same value sizes MGARD-X's outlier buffers.
inline double OutlierRatio() {
    static const double v = [] { const char* e = std::getenv("HPMGARD_XHUFF_OUTLIER"); const double x = e ? std::atof(e) : 0.01; return x >= 0.0 && x <= 1.0 ? x : 0.01; }();
    return v;
}
// Streams shorter than this use the in-house Huffman (MGARD-X's serialised codebook costs ~4 bytes per dictionary entry).
constexpr size_t kMinSymbols = size_t(1) << 16;
// Dictionary actually used for a stream: the smallest power of two above its largest zigzag code, at least
// 256 and at most DictSize(). MGARD-X serialises ~4 bytes per entry, so a small alphabet should not pay
// for 8192 entries (34 KB per stream); the chosen size is recorded in the payload.
inline uint32_t ChooseDict(uint32_t zmax) {
    uint32_t d = 256; while (d <= zmax) d <<= 1;
    return d > DictSize() ? DictSize() : d;
}

bool Available(int device);               // 0 = SERIAL (needs MGARD-X), 1 = HIP (needs the HIP build + a device)
bool AlphabetOk(XType t, uint32_t zmin, uint32_t zmax);   // false when the type cannot code this code range

// Payload written by Encode*: u32 dict, u32 block (little endian), then the MGARD-X stream.
// level: zstd level for XType::HuffmanZstd (ignored otherwise). All return false when MGARD-X cannot code
// the stream (code >= dict, n < kMinSymbols, MGARD-X failure, device error, failed verification); the
// caller then falls back to the in-house coder.
bool Encode(int device, const int32_t* sym, size_t n, XType t, int level, std::vector<uint8_t>& out);
// Symbols already in device memory (device 1 = HIP). No host copy of the symbols is made.
bool EncodeDevice(int device, const int32_t* d_sym, size_t n, XType t, int level, std::vector<uint8_t>& out);
bool Decode(int device, const uint8_t* b, size_t nb, int32_t* out, size_t n, XType t, int level);
// Decode on the device and leave the symbols there (pointer valid until the next call of this stage on
// that device); nullptr when not possible (CPU device, MGARD-X unavailable, failure).
const int32_t* DecodeDevice(int device, const uint8_t* b, size_t nb, size_t n, XType t, int level);
// Loads the device kernels and allocates the MGARD-X workspace once per process per (device, type) so the
// first real stream does not pay for it. No-op on the CPU and when MGARD-X is unavailable.
void Warmup(int device, XType t);

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
