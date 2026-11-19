// SERIAL (CPU) instantiation of the MGARD-X lossless stage. Compiled as C++ WITHOUT OpenMP (own target,
// see CMakeLists.txt) because MGARD-X declares serial_dev_id '#pragma omp threadprivate' while libmgard
// defines a plain int.
#include "hpmgard/codec/MgardXHuffImpl.hpp"

namespace hpmgard {
namespace codec {
namespace xhuff {

template <> bool DeviceErrorCheck<mgard_x::SERIAL>() { return true; }

bool EncodeSerial(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) { return Impl<mgard_x::SERIAL>::Encode(codes, t, level, dict, block, out); }
bool DecodeSerial(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block) { return Impl<mgard_x::SERIAL>::Decode(b, nb, codes, n, t, level, dict, block); }

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
