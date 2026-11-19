// MGARD-X lossless (Huffman, optionally + Zstd inside MGARD-X) as a first-stage symbol coder, running
// on the SERIAL (CPU) or HIP device. The stream is device-agnostic (MGARD-X's Huffman serialisation
// and Zstd are the same on every backend), so a GPU-coded stream decodes on the CPU and vice versa.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

constexpr uint32_t kDictSize = 8192;      // MGARD-X Huffman dictionary (codes must stay below it)
constexpr uint32_t kBlockSize = 20480;    // MGARD-X Huffman block size

bool Available(int device);               // 0 = SERIAL (needs MGARD-X), 1 = HIP (needs the HIP build + a device)

// Zigzag-maps the symbols and codes them with ComposedLosslessCompressor<int32_t, uint64_t, Device>.
// zstd_level 0 = Huffman only, otherwise Huffman_Zstd at that level. Returns false when MGARD-X cannot
// code the stream (a code >= kDictSize, or an MGARD-X failure); the caller then falls back.
bool Encode(int device, const int32_t* sym, size_t n, int zstd_level, std::vector<uint8_t>& out);
bool Decode(int device, const uint8_t* b, size_t nb, int32_t* out, size_t n, int zstd_level);

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
