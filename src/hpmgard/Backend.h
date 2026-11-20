// Device backend of the per-component numeric stages. The compressor drives one of these; the codecs
// and the header are device-independent. GPU backends keep the component resident across the stages,
// hand the quantised residual symbols to a device-capable codec without a host round trip, and give the
// compressor pinned host scratch for the arrays that cross the PCIe/IF link.
#pragma once
#include "hpmgard/BaseSplit.h"
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

    // ---- device-resident symbols -------------------------------------------------------------------
    // True when ResidualFromBase may be called with k == nullptr: the quantised symbols then stay on the
    // device (DeviceSymbols()) and are fetched with CopySymbolsToHost only if a host coder needs them.
    virtual bool KeepsDeviceSymbols() const { return false; }
    // Device pointer to the symbols of the last ResidualFromBase/Quantize call (valid until the next call
    // on this backend); nullptr on the CPU.
    virtual const int32_t* DeviceSymbols() const { return nullptr; }
    virtual void CopySymbolsToHost(int32_t* /*k*/) const {}
    // ---- device-resident base path (GPU backends) ---------------------------------------------------
    // Restrict -> split (cell average / dedup / plain) -> quantise -> decoded base, all on the device; the
    // quantised base streams are exposed as device pointers (valid until the next base call) and the decoded
    // base stays resident for ResidualFromBase(base_dec == nullptr). False = not supported, use the host path.
    struct DeviceStreams { const int32_t* d_avg = nullptr; size_t n_avg = 0; const int32_t* d_rem = nullptr; size_t n_rem = 0; };
    virtual bool SupportsDeviceBase() const { return false; }
    virtual bool RestrictDevice(const OperatorSet& /*ops*/, size_t /*ncells*/, const double* /*u*/) { return false; }
    // Returns false on quantiser overflow (caller falls back to raw) or when unsupported.
    virtual bool SplitCellAvgDevice(size_t /*ncells*/, int /*nb*/, double /*qb*/, DeviceStreams& /*out*/) { return false; }
    virtual bool SplitDedupDevice(size_t /*ncells*/, int /*nb*/, double /*qb*/, const DedupGeometry& /*geom*/, DeviceStreams& /*out*/) { return false; }
    virtual bool SplitPlainDevice(size_t /*ncells*/, int /*nb*/, double /*qb*/, DeviceStreams& /*out*/) { return false; }
    // Copies n symbols from a device pointer returned above to the host (host coders, fallbacks).
    virtual void CopyDeviceToHost(const int32_t* /*d*/, size_t /*n*/, int32_t* /*host*/) const {}

    // k = rint((u - P base_dec) / q) without materialising P base_dec on the host (same overflow rule).
    // k may be nullptr only when KeepsDeviceSymbols(); base_dec may be nullptr after a Split*Device call
    // (the resident decoded base is used).
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
    // Same with the symbols already in device memory (e.g. codec::DecodeSymbolsToDevice); false = not supported here.
    virtual bool ReconstructFromDeviceSymbols(const OperatorSet& /*ops*/, const double* /*base_dec*/, size_t /*ncells*/, const int32_t* /*d_k*/, double /*q*/, double* /*out*/) { return false; }

    // ---- host scratch --------------------------------------------------------------------------------
    // Grow-only host buffers for the component arrays that the backend moves to or from the device
    // (pinned on GPU backends so the copies run at link speed). One buffer per slot, valid until the next
    // call with the same slot on this backend; contents are not preserved across a growth.
    enum HostSlot { SlotU = 0, SlotK = 1, SlotBaseDec = 2, SlotOut = 3, SlotBase = 4, NumHostSlots = 5 };
    virtual void* HostScratch(int slot, size_t bytes) {
        Scratch& s = host_scratch_[slot];
        if (s.bytes < bytes) { s.p.reset(new char[bytes]); s.bytes = bytes; }
        return s.p.get();
    }
private:
    struct Scratch { std::unique_ptr<char[]> p; size_t bytes = 0; };
    Scratch host_scratch_[NumHostSlots];
};

template <class T> inline T* HostArray(Backend& be, int slot, size_t n) { return static_cast<T*>(be.HostScratch(slot, n * sizeof(T))); }

constexpr double kMaxQuantMagnitude = 1.0e9;

DeviceType ParseDevice(const std::string& s);        // "auto" | "cpu" | "hip" | "cuda"
const char* DeviceName(DeviceType t);
bool DeviceAvailable(DeviceType t);
// Picks the requested backend; `auto` = HIP if built and a device is present, else CUDA likewise, else CPU.
// GPU backends are process-wide singletons (device buffers, pinned scratch and uploaded operators persist
// across calls); they are not thread safe.
std::unique_ptr<Backend> MakeBackend(const std::string& requested);

} // namespace hpmgard
