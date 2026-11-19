// MGARD-X lossless (Huffman, optionally + Zstd inside MGARD-X) as a first-stage symbol coder, running
// on the SERIAL (CPU) or HIP device. The stream is device-agnostic (MGARD-X's Huffman serialisation
// and Zstd are the same on every backend), so a GPU-coded stream decodes on the CPU and vice versa.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

// MGARD-X Huffman dictionary (zigzag codes must stay below it) and block size. Defaults 8192 / 20480;
// override with HPMGARD_XHUFF_DICT / HPMGARD_XHUFF_BLOCK (read once). Both are recorded in the stream.
inline uint32_t DictSize() {
    static const uint32_t v = [] { const char* e = std::getenv("HPMGARD_XHUFF_DICT"); const long x = e ? std::atol(e) : 0; return x >= 2 && x <= 65536 ? static_cast<uint32_t>(x) : 8192u; }();
    return v;
}
inline uint32_t BlockSize() {
    static const uint32_t v = [] { const char* e = std::getenv("HPMGARD_XHUFF_BLOCK"); const long x = e ? std::atol(e) : 0; return x >= 64 && x <= (1l << 24) ? static_cast<uint32_t>(x) : 20480u; }();
    return v;
}

bool Available(int device);               // 0 = SERIAL (needs MGARD-X), 1 = HIP (needs the HIP build + a device)

// Zigzag-maps the symbols and codes them with ComposedLosslessCompressor<int32_t, uint64_t, Device>.
// zstd_level 0 = Huffman only, otherwise Huffman_Zstd at that level. Returns false when MGARD-X cannot
// code the stream (a code >= kDictSize, or an MGARD-X failure); the caller then falls back.
// The payload starts with u32 dict, u32 block (little endian), then the MGARD-X stream.
bool Encode(int device, const int32_t* sym, size_t n, int zstd_level, std::vector<uint8_t>& out);
bool Decode(int device, const uint8_t* b, size_t nb, int32_t* out, size_t n, int zstd_level);

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
