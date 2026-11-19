// HIP instantiation of the MGARD-X Huffman stage (compiled as HIP with MGARDX_COMPILE_HIP).
#include "hpmgard/codec/MgardXHuffImpl.hpp"
#include <hip/hip_runtime.h>

namespace hpmgard {
namespace codec {
namespace xhuff {

bool HipDevicePresent() { int count = 0; return hipGetDeviceCount(&count) == hipSuccess && count > 0; }
bool EncodeHip(const std::vector<int32_t>& codes, int zstd_level, std::vector<uint8_t>& out) { return Impl<mgard_x::HIP>::Encode(codes, zstd_level, out); }
bool DecodeHip(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level) { return Impl<mgard_x::HIP>::Decode(b, nb, codes, n, zstd_level); }

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
