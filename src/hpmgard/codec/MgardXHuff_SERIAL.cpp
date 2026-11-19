// SERIAL (CPU) instantiation of the MGARD-X Huffman stage. Compiled as C++ (falls through to
// MGARDX_COMPILE_SERIAL in MGARD-X's headers).
#include "hpmgard/codec/MgardXHuffImpl.hpp"

namespace hpmgard {
namespace codec {
namespace xhuff {

template <> bool DeviceErrorCheck<mgard_x::SERIAL>() { return true; }

bool EncodeSerial(const std::vector<int32_t>& codes, int zstd_level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) { return Impl<mgard_x::SERIAL>::Encode(codes, zstd_level, dict, block, out); }
bool DecodeSerial(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level, uint32_t dict, uint32_t block) { return Impl<mgard_x::SERIAL>::Decode(b, nb, codes, n, zstd_level, dict, block); }

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
