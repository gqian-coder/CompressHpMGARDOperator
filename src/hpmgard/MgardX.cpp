#include "hpmgard/MgardX.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef HPMGARD_HAVE_MGARDX
#include <mgard/compress_x.hpp>
#include <mgard/MGARDConfig.hpp>
#include <mgard/mgard-x/RuntimeX/MGARDXConfig.h>
#endif

namespace hpmgard {
namespace mgardx {

#ifdef HPMGARD_HAVE_MGARDX

bool Available() { return true; }

void Version(int& major, int& minor, int& patch) { major = MGARD_VERSION_MAJOR; minor = MGARD_VERSION_MINOR; patch = MGARD_VERSION_PATCH; }

namespace {
mgard_x::Config MakeConfig(int levels, int device) {
    mgard_x::Config cfg;
    cfg.dev_type = mgard_x::device_type::SERIAL;
#if MGARD_ENABLE_HIP
    if (device == 1) cfg.dev_type = mgard_x::device_type::HIP;
#endif
#if MGARD_ENABLE_CUDA
    if (device == 2) cfg.dev_type = mgard_x::device_type::CUDA;
#endif
    cfg.lossless = mgard_x::lossless_type::Huffman_Zstd;
    cfg.log_level = 0;
    if (levels > 0) cfg.num_global_refactoring_level = static_cast<mgard_x::SIZE>(levels);
    return cfg;
}
}

std::vector<uint8_t> Compress1D(const double* data, size_t n, double abs_tol, double s, int levels, int device) {
    mgard_x::Config cfg = MakeConfig(levels, device);
    std::vector<mgard_x::SIZE> shape{static_cast<mgard_x::SIZE>(n)};
    void* out = nullptr; size_t out_size = 0;
    const auto st = mgard_x::compress(1, mgard_x::data_type::Double, shape, abs_tol, s, mgard_x::error_bound_type::ABS,
                                      data, out, out_size, cfg, false);
    if (st != mgard_x::compress_status_type::Success) throw std::runtime_error("mgard_x::compress failed with status " + std::to_string(static_cast<int>(st)));
    std::vector<uint8_t> v(static_cast<uint8_t*>(out), static_cast<uint8_t*>(out) + out_size);
    std::free(out);
    return v;
}

void Decompress(const uint8_t* b, size_t nb, std::vector<double>& out, int device) {
    mgard_x::Config cfg = MakeConfig(0, device);
    void* dec = nullptr; std::vector<mgard_x::SIZE> shape; mgard_x::data_type dt;
    const auto st = mgard_x::decompress(b, nb, dec, shape, dt, cfg, false);
    if (st != mgard_x::compress_status_type::Success) throw std::runtime_error("mgard_x::decompress failed with status " + std::to_string(static_cast<int>(st)));
    if (dt != mgard_x::data_type::Double) { std::free(dec); throw std::runtime_error("mgard_x: unexpected data type"); }
    size_t n = 1; for (auto d : shape) n *= d;
    out.assign(static_cast<double*>(dec), static_cast<double*>(dec) + n);
    std::free(dec);
}

bool DeviceWorks(int device) {
    static int cache[3] = {-1, -1, -1};
    if (device < 0 || device > 2) return false;
    if (cache[device] >= 0) return cache[device] == 1;
    bool ok = false;
    try {
        const size_t n = 4096; std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) x[i] = std::sin(0.01 * static_cast<double>(i)) + 1e-3 * static_cast<double>(i % 7);
        const double tol = 1e-3;
        const std::vector<uint8_t> b = Compress1D(x.data(), n, tol, std::numeric_limits<double>::infinity(), 0, device);
        std::vector<double> y; Decompress(b.data(), b.size(), y, device);
        ok = (y.size() == n);
        for (size_t i = 0; ok && i < n; ++i) if (!(std::fabs(y[i] - x[i]) <= tol * 1.0001)) ok = false;
    } catch (...) { ok = false; }
    cache[device] = ok ? 1 : 0;
    return ok;
}

#else

bool Available() { return false; }
void Version(int& major, int& minor, int& patch) { major = minor = patch = 0; }
std::vector<uint8_t> Compress1D(const double*, size_t, double, double, int, int) { throw std::runtime_error("hpMGARD was built without MGARD-X"); }
void Decompress(const uint8_t*, size_t, std::vector<double>&, int) { throw std::runtime_error("hpMGARD was built without MGARD-X"); }
bool DeviceWorks(int) { return false; }

#endif

} // namespace mgardx
} // namespace hpmgard
