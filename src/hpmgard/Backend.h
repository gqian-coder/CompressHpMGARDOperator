// Device backend of the per-component numeric stages. The compressor drives one of these; the codecs
// and the header are device-independent. v1: every call takes host arrays and the GPU backends stage
// them to the device internally (correct first, resident-data optimisation later).
#pragma once
#include "hpmgard/Operators.h"
#include "hpmgard/Types.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

    // ---- resident-data session (GPU backends keep the component on the device between the calls) ----
    // Declares `u` (npts values) as the component being processed; Restrict/ResidualFromBase on the same
    // pointer reuse the device copy. CPU backend: no-op.
    virtual void BeginComponent(const double* /*u*/, size_t /*npts*/) {}
    virtual void EndComponent() {}
    // Device pointer to the quantised symbols produced by the last ResidualFromBase/Quantize call (valid
    // until the next call on this backend); nullptr on the CPU. Lets a device-capable codec code them
    // without a host round trip.
    virtual const int32_t* DeviceSymbols() const { return nullptr; }
    // k = rint((u - P base_dec) / q) without materialising P base_dec on the host (same overflow rule).
    virtual bool ResidualFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, double q, int32_t* k, const double* u) {
        std::vector<double> ub(ncells * static_cast<size_t>(ops.nsp_data));
        Prolong(ops, base_dec, ncells, ub.data());
        return ResidualQuantize(u, ub.data(), ub.size(), q, k);
    }
    // out = P base_dec + k * q
    virtual void ReconstructFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, const int32_t* k, double q, double* out) {
        Prolong(ops, base_dec, ncells, out);
        DequantizeAdd(k, ncells * static_cast<size_t>(ops.nsp_data), q, out);
    }
};

constexpr double kMaxQuantMagnitude = 1.0e9;

DeviceType ParseDevice(const std::string& s);        // "auto" | "cpu" | "hip" | "cuda"
const char* DeviceName(DeviceType t);
bool DeviceAvailable(DeviceType t);
// Picks the requested backend; `auto` = HIP if built and a device is present, else CUDA likewise, else CPU.
std::unique_ptr<Backend> MakeBackend(const std::string& requested);

} // namespace hpmgard
