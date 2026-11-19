// HIP instantiation of the MGARD-X Huffman stage (compiled as HIP with MGARDX_COMPILE_HIP).
#include "hpmgard/codec/MgardXHuffImpl.hpp"
#include <hip/hip_runtime.h>

namespace hpmgard {
namespace codec {
namespace xhuff {

template <> bool DeviceErrorCheck<mgard_x::HIP>() {
    const hipError_t sync = hipDeviceSynchronize();
    const hipError_t last = hipGetLastError();
    if (sync == hipSuccess && last == hipSuccess) return true;
    std::cerr << "[hpmgard xhuff] HIP error: " << hipGetErrorString(sync != hipSuccess ? sync : last) << "\n";
    return false;
}

bool HipDevicePresent() { int count = 0; return hipGetDeviceCount(&count) == hipSuccess && count > 0; }
bool EncodeHip(const std::vector<int32_t>& codes, int zstd_level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) { return Impl<mgard_x::HIP>::Encode(codes, zstd_level, dict, block, out); }
bool DecodeHip(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level, uint32_t dict, uint32_t block) { return Impl<mgard_x::HIP>::Decode(b, nb, codes, n, zstd_level, dict, block); }

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
