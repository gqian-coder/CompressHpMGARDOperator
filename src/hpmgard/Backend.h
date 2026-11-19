// Device backend of the per-component numeric stages. The compressor drives one of these; the codecs
// and the header are device-independent. v1: every call takes host arrays and the GPU backends stage
// them to the device internally (correct first, resident-data optimisation later).
#pragma once
#include "hpmgard/Operators.h"
#include "hpmgard/Types.h"
#include <cstdint>
#include <memory>
#include <string>

namespace hpmgard {

enum class DeviceType : uint8_t { CPU = 0, HIP = 1, CUDA = 2 };

class Backend {
public:
    virtual ~Backend() = default;
    virtual DeviceType type() const = 0;
    virtual const char* name() const = 0;
    // base = R u (per cell), u = P base (per cell); u: [ncells][nsp_data], base: [ncells][nsp_base]
    virtual void Restrict(const OperatorSet& ops, const double* u, size_t ncells, double* base) = 0;
    virtual void Prolong(const OperatorSet& ops, const double* base, size_t ncells, double* u) = 0;
    // k = rint(x / q); returns false if any |x / q| exceeds the int32 safety limit (then k is unusable)
    virtual bool Quantize(const double* x, size_t n, double q, int32_t* k) = 0;
    // k = rint((u - ub) / q) with the same overflow rule
    virtual bool ResidualQuantize(const double* u, const double* ub, size_t n, double q, int32_t* k) = 0;
    // u += k * q
    virtual void DequantizeAdd(const int32_t* k, size_t n, double q, double* u) = 0;
    // avg[cell] = mean of the nb base values of the cell
    virtual void CellMeans(const double* base, size_t ncells, int nb, double* avg) = 0;
};

constexpr double kMaxQuantMagnitude = 1.0e9;

DeviceType ParseDevice(const std::string& s);        // "auto" | "cpu" | "hip" | "cuda"
const char* DeviceName(DeviceType t);
bool DeviceAvailable(DeviceType t);
// Picks the requested backend; `auto` = HIP if built and a device is present, else CUDA likewise, else CPU.
std::unique_ptr<Backend> MakeBackend(const std::string& requested);

} // namespace hpmgard
