#include "hpmgard/MgardX.h"
#include <stdexcept>
#include <string>

#ifdef HPMGARD_HAVE_MGARDX
#include <mgard/compress_x.hpp>
#include <mgard/MGARDConfig.hpp>
#endif

namespace hpmgard {
namespace mgardx {

#ifdef HPMGARD_HAVE_MGARDX

bool Available() { return true; }

void Version(int& major, int& minor, int& patch) { major = MGARD_VERSION_MAJOR; minor = MGARD_VERSION_MINOR; patch = MGARD_VERSION_PATCH; }

namespace {
mgard_x::Config MakeConfig(int levels) {
    mgard_x::Config cfg;
#if defined(HPMGARD_MGARDX_DEVICE_HIP)
    cfg.dev_type = mgard_x::device_type::HIP;
#elif defined(HPMGARD_MGARDX_DEVICE_CUDA)
    cfg.dev_type = mgard_x::device_type::CUDA;
#else
    cfg.dev_type = mgard_x::device_type::SERIAL;
#endif
    cfg.lossless = mgard_x::lossless_type::Huffman_Zstd;
    cfg.log_level = 0;
    if (levels > 0) cfg.num_global_refactoring_level = static_cast<mgard_x::SIZE>(levels);
    return cfg;
}
}

std::vector<uint8_t> Compress1D(const double* data, size_t n, double abs_tol, double s, int levels) {
    mgard_x::Config cfg = MakeConfig(levels);
    std::vector<mgard_x::SIZE> shape{static_cast<mgard_x::SIZE>(n)};
    void* out = nullptr; size_t out_size = 0;
    const auto st = mgard_x::compress(1, mgard_x::data_type::Double, shape, abs_tol, s, mgard_x::error_bound_type::ABS,
                                      data, out, out_size, cfg, false);
    if (st != mgard_x::compress_status_type::Success) throw std::runtime_error("mgard_x::compress failed with status " + std::to_string(static_cast<int>(st)));
    std::vector<uint8_t> v(static_cast<uint8_t*>(out), static_cast<uint8_t*>(out) + out_size);
    std::free(out);
    return v;
}

void Decompress(const uint8_t* b, size_t nb, std::vector<double>& out) {
    mgard_x::Config cfg = MakeConfig(0);
    void* dec = nullptr; std::vector<mgard_x::SIZE> shape; mgard_x::data_type dt;
    const auto st = mgard_x::decompress(b, nb, dec, shape, dt, cfg, false);
    if (st != mgard_x::compress_status_type::Success) throw std::runtime_error("mgard_x::decompress failed with status " + std::to_string(static_cast<int>(st)));
    if (dt != mgard_x::data_type::Double) { std::free(dec); throw std::runtime_error("mgard_x: unexpected data type"); }
    size_t n = 1; for (auto d : shape) n *= d;
    out.assign(static_cast<double*>(dec), static_cast<double*>(dec) + n);
    std::free(dec);
}

#else

bool Available() { return false; }
void Version(int& major, int& minor, int& patch) { major = minor = patch = 0; }
std::vector<uint8_t> Compress1D(const double*, size_t, double, double, int) { throw std::runtime_error("hpMGARD was built without MGARD-X"); }
void Decompress(const uint8_t*, size_t, std::vector<double>&) { throw std::runtime_error("hpMGARD was built without MGARD-X"); }

#endif

} // namespace mgardx
} // namespace hpmgard
