#include "hpmgard/codec/Frbp.h"

namespace hpmgard {
namespace codec {
namespace frbp {

#define HPM_FRBP_DECLS \
    bool DevicePresent(); \
    bool EncodeDevice(const int32_t* d_sym, size_t n, double eb_info, std::vector<uint8_t>& out); \
    const int32_t* DecodeDevice(const uint8_t* b, size_t nb, size_t n);
#ifdef HPMGARD_HAVE_HIP
namespace hip { HPM_FRBP_DECLS }
#endif
#ifdef HPMGARD_HAVE_CUDA
namespace cuda { HPM_FRBP_DECLS }
#endif
#undef HPM_FRBP_DECLS

bool Available(int device) {
#ifdef HPMGARD_HAVE_HIP
    if (device == 1) return hip::DevicePresent();
#endif
#ifdef HPMGARD_HAVE_CUDA
    if (device == 2) return cuda::DevicePresent();
#endif
    (void)device;
    return false;
}

bool EncodeDevice(int device, const int32_t* d_sym, size_t n, double eb_info, std::vector<uint8_t>& out) {
    if (!d_sym || !Available(device)) return false;
#ifdef HPMGARD_HAVE_HIP
    if (device == 1) return hip::EncodeDevice(d_sym, n, eb_info, out);
#endif
#ifdef HPMGARD_HAVE_CUDA
    if (device == 2) return cuda::EncodeDevice(d_sym, n, eb_info, out);
#endif
    (void)n; (void)eb_info; (void)out;
    return false;
}

const int32_t* DecodeDevice(int device, const uint8_t* b, size_t nb, size_t n) {
    if (!Available(device)) return nullptr;
#ifdef HPMGARD_HAVE_HIP
    if (device == 1) return hip::DecodeDevice(b, nb, n);
#endif
#ifdef HPMGARD_HAVE_CUDA
    if (device == 2) return cuda::DecodeDevice(b, nb, n);
#endif
    (void)b; (void)nb; (void)n;
    return nullptr;
}

} // namespace frbp
} // namespace codec
} // namespace hpmgard
