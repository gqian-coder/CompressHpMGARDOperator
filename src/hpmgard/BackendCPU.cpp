#include "hpmgard/Backend.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace hpmgard {

namespace {
class CpuBackend : public Backend {
public:
    DeviceType type() const override { return DeviceType::CPU; }
    const char* name() const override { return "cpu"; }
    void Restrict(const OperatorSet& ops, const double* u, size_t ncells, double* base) override { ops.Restrict(u, ncells, base); }
    void Prolong(const OperatorSet& ops, const double* base, size_t ncells, double* u) override { ops.Prolong(base, ncells, u); }
    bool Quantize(const double* x, size_t n, double q, int32_t* k) override {
        bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok)
        for (size_t i = 0; i < n; ++i) {
            const double v = x[i] / q;
            if (!(std::fabs(v) < kMaxQuantMagnitude)) ok = false;
            k[i] = static_cast<int32_t>(std::llrint(std::fabs(v) < kMaxQuantMagnitude ? v : 0.0));
        }
        return ok;
    }
    bool ResidualQuantize(const double* u, const double* ub, size_t n, double q, int32_t* k) override {
        bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok)
        for (size_t i = 0; i < n; ++i) {
            const double v = (u[i] - ub[i]) / q;
            if (!(std::fabs(v) < kMaxQuantMagnitude)) ok = false;
            k[i] = static_cast<int32_t>(std::llrint(std::fabs(v) < kMaxQuantMagnitude ? v : 0.0));
        }
        return ok;
    }
    void DequantizeAdd(const int32_t* k, size_t n, double q, double* u) override {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) u[i] += k[i] * q;
    }
    void CellMeans(const double* base, size_t ncells, int nb, double* avg) override {
#pragma omp parallel for schedule(static)
        for (size_t c = 0; c < ncells; ++c) { double s = 0.0; for (int j = 0; j < nb; ++j) s += base[c * nb + j]; avg[c] = s / nb; }
    }
};
}

std::unique_ptr<Backend> MakeHipBackend();   // BackendHIP.hip (nullptr when not built or no device)

DeviceType ParseDevice(const std::string& in) {
    std::string s; for (char c : in) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "cpu" || s == "serial" || s == "host") return DeviceType::CPU;
    if (s == "hip" || s == "gpu" || s == "amd") return DeviceType::HIP;
    if (s == "cuda" || s == "nvidia") return DeviceType::CUDA;
    if (s == "auto" || s.empty()) {
        if (DeviceAvailable(DeviceType::HIP)) return DeviceType::HIP;
        if (DeviceAvailable(DeviceType::CUDA)) return DeviceType::CUDA;
        return DeviceType::CPU;
    }
    throw std::invalid_argument("unknown device '" + in + "' (auto|cpu|hip|cuda)");
}

const char* DeviceName(DeviceType t) { return t == DeviceType::CPU ? "cpu" : t == DeviceType::HIP ? "hip" : "cuda"; }

bool DeviceAvailable(DeviceType t) {
    if (t == DeviceType::CPU) return true;
    if (t == DeviceType::HIP) { auto b = MakeHipBackend(); return b != nullptr; }
    return false;   // CUDA backend not built yet
}

std::unique_ptr<Backend> MakeBackend(const std::string& requested) {
    const DeviceType t = ParseDevice(requested);
    if (t == DeviceType::HIP) { auto b = MakeHipBackend(); if (!b) throw std::runtime_error("HIP backend requested but not available (not built or no device)"); return b; }
    if (t == DeviceType::CUDA) throw std::runtime_error("CUDA backend not implemented yet");
    return std::unique_ptr<Backend>(new CpuBackend());
}

#ifndef HPMGARD_HAVE_HIP
std::unique_ptr<Backend> MakeHipBackend() { return nullptr; }
#endif

} // namespace hpmgard
