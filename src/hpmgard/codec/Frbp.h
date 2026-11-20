// GPU implementation of the FRBP stage (same bytes as the CPU FrbpEncode/FrbpDecode in RawFrbpZstd.cpp):
// fixed-rate bit packing of 32-value blocks, one rate byte per block, sign word + `rate` bit planes per
// non-zero block. Encode reads the quantised symbols from device memory and returns the host stream;
// decode takes the host stream and leaves the symbols on the device.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hpmgard {
namespace codec {
namespace frbp {

bool Available(int device);                                                       // 1 = HIP, 2 = CUDA (built + device present)
bool EncodeDevice(int device, const int32_t* d_sym, size_t n, double eb_info, std::vector<uint8_t>& out);
// Device pointer to the decoded symbols (valid until the next call of this stage on that device); nullptr on failure.
const int32_t* DecodeDevice(int device, const uint8_t* b, size_t nb, size_t n);

} // namespace frbp
} // namespace codec
} // namespace hpmgard
