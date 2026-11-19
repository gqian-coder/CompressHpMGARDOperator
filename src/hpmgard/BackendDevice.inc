// GPU backend shared by HIP and CUDA (included by BackendHIP.hip / BackendCUDA.cu after DeviceShim.h).
// - tensor shapes: per-cell tensor-product sweeps (one thread per cell, same summation order as the CPU
//   ApplyTensor), simplex shapes: dense per-cell matrix
// - the component stays resident on the device across Restrict / ResidualFromBase; the quantised
//   residual symbols stay on the device for a device-capable codec (DeviceSymbols) and a device-decoded
//   symbol array feeds the reconstruction directly (ReconstructFromDeviceSymbols)
// - pinned host scratch for the arrays that cross the link
// - one process-wide instance per device type (buffers and the uploaded operator persist across calls)
#include "hpmgard/Backend.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpmgard {

namespace {

#define HPM_DEV_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) throw std::runtime_error(std::string(HPM_DEVICE_NAME " error: ") + hipGetErrorString(e_) + " at " #call); } while (0)

constexpr int kThreads = 256;
inline unsigned GridFor(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

__global__ void DenseApplyKernel(const double* __restrict__ M, int n_out, int n_in, const double* __restrict__ in, double* __restrict__ out, size_t ncells) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t total = ncells * (size_t)n_out;
    if (idx >= total) return;
    const size_t c = idx / n_out; const int a = static_cast<int>(idx - c * n_out);
    const double* x = in + c * (size_t)n_in; const double* m = M + (size_t)a * n_in;
    double s = 0.0;
    for (int i = 0; i < n_in; ++i) s += m[i] * x[i];
    out[idx] = s;
}

// 3-D tensor-product apply, one thread per cell, intermediates in registers/local memory. Mirrors the
// loop order of the CPU ApplyTensor (axis i, then j, then k; i fastest) so both backends sum in the
// same order. NMAX >= max(n_in*n_in*n_out, n_in*n_out*n_out).
template <int NMAX>
__global__ void Tensor3Kernel(const double* __restrict__ M, int n_out, int n_in, const double* __restrict__ in, double* __restrict__ out, size_t ncells) {
    const size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    const int nin3 = n_in * n_in * n_in, nout3 = n_out * n_out * n_out;
    const double* x = in + c * (size_t)nin3;
    double* y = out + c * (size_t)nout3;
    double t1[NMAX], t2[NMAX];
    // axis i: t1[(k*n_in + j)*n_out + a]
    for (int kj = 0; kj < n_in * n_in; ++kj) {
        const double* xs = x + kj * n_in;
        for (int a = 0; a < n_out; ++a) {
            double s = 0.0; const double* m = M + a * n_in;
            for (int i = 0; i < n_in; ++i) s += m[i] * xs[i];
            t1[kj * n_out + a] = s;
        }
    }
    // axis j: t2[(k*n_out + b)*n_out + a]
    for (int k = 0; k < n_in; ++k)
        for (int b = 0; b < n_out; ++b) {
            const double* m = M + b * n_in;
            double* ts = t2 + (k * n_out + b) * n_out;
            for (int a = 0; a < n_out; ++a) ts[a] = 0.0;
            for (int j = 0; j < n_in; ++j) {
                const double mj = m[j];
                const double* t1s = t1 + (k * n_in + j) * n_out;
                for (int a = 0; a < n_out; ++a) ts[a] += mj * t1s[a];
            }
        }
    // axis k: y[(cc*n_out + b)*n_out + a]
    for (int cc = 0; cc < n_out; ++cc) {
        const double* m = M + cc * n_in;
        double* ys = y + cc * n_out * n_out;
        for (int ba = 0; ba < n_out * n_out; ++ba) ys[ba] = 0.0;
        for (int k = 0; k < n_in; ++k) {
            const double mk = m[k];
            const double* t2s = t2 + k * n_out * n_out;
            for (int ba = 0; ba < n_out * n_out; ++ba) ys[ba] += mk * t2s[ba];
        }
    }
}

template <int NMAX>
__global__ void Tensor2Kernel(const double* __restrict__ M, int n_out, int n_in, const double* __restrict__ in, double* __restrict__ out, size_t ncells) {
    const size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    const double* x = in + c * (size_t)(n_in * n_in);
    double* y = out + c * (size_t)(n_out * n_out);
    double t1[NMAX];
    for (int j = 0; j < n_in; ++j)
        for (int a = 0; a < n_out; ++a) {
            double s = 0.0; const double* m = M + a * n_in;
            for (int i = 0; i < n_in; ++i) s += m[i] * x[j * n_in + i];
            t1[j * n_out + a] = s;
        }
    for (int b = 0; b < n_out; ++b) {
        const double* m = M + b * n_in;
        for (int a = 0; a < n_out; ++a) {
            double s = 0.0;
            for (int j = 0; j < n_in; ++j) s += m[j] * t1[j * n_out + a];
            y[b * n_out + a] = s;
        }
    }
}

__global__ void QuantizeKernel(const double* __restrict__ x, const double* __restrict__ sub, size_t n, double inv_q, int32_t* __restrict__ k, int* __restrict__ overflow) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double v = (sub ? (x[i] - sub[i]) : x[i]) * inv_q;
    if (!(fabs(v) < kMaxQuantMagnitude)) { atomicOr(overflow, 1); k[i] = 0; return; }
    k[i] = static_cast<int32_t>(llrint(v));
}

__global__ void DequantizeAddKernel(const int32_t* __restrict__ k, size_t n, double q, double* __restrict__ u) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) u[i] += k[i] * q;
}

__global__ void CellMeansKernel(const double* __restrict__ base, size_t ncells, int nb, double* __restrict__ avg) {
    const size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    double s = 0.0; for (int j = 0; j < nb; ++j) s += base[c * (size_t)nb + j];
    avg[c] = s / nb;
}

struct DeviceBuffer {
    void* p = nullptr; size_t bytes = 0;
    void reserve(size_t b) { if (b > bytes) { if (p) (void)hipFree(p); HPM_DEV_CHECK(hipMalloc(&p, b)); bytes = b; } }
    // never freed in a destructor: the instance lives for the process and the runtime may be gone at exit
};
struct PinnedBuffer {
    void* p = nullptr; size_t bytes = 0;
    void reserve(size_t b) { if (b > bytes) { if (p) (void)hipHostFree(p); HPM_DEV_CHECK(hipHostMalloc(&p, b, hipHostMallocDefault)); bytes = b; } }
};

class DeviceBackend : public Backend {
public:
    explicit DeviceBackend(int device) { HPM_DEV_CHECK(hipSetDevice(device)); }
    DeviceType type() const override { return HPM_DEVICE_TYPE; }
    const char* name() const override { return HPM_DEVICE_NAME; }

    // ---- operator upload (cached by content) and apply -----------------------------------------------
    void UploadOperator(const OperatorSet& ops, bool restrict) {
        const std::vector<double>& M = ops.tensor ? (restrict ? ops.RE1 : ops.Pc1) : (restrict ? ops.Rfull : ops.Pfull);
        if (M == op_cache_) return;
        d_m_.reserve(M.size() * 8);
        HPM_DEV_CHECK(hipMemcpy(d_m_.p, M.data(), M.size() * 8, hipMemcpyHostToDevice));
        op_cache_ = M;
    }
    // in: [ncells][n_in^dim] -> out: [ncells][n_out^dim] for tensor shapes (n_* = 1-D sizes), dense otherwise
    void Apply(const OperatorSet& ops, bool restrict, const void* d_in, size_t ncells, void* d_out) {
        UploadOperator(ops, restrict);
        const double* M = static_cast<const double*>(d_m_.p);
        const int nsp_in = restrict ? ops.nsp_data : ops.nsp_base, nsp_out = restrict ? ops.nsp_base : ops.nsp_data;
        if (!ops.tensor) {
            const size_t nout = ncells * nsp_out;
            hipLaunchKernelGGL(DenseApplyKernel, dim3(GridFor(nout)), dim3(kThreads), 0, 0, M, nsp_out, nsp_in, (const double*)d_in, (double*)d_out, ncells);
        } else {
            const int n_in = restrict ? ops.p + 1 : ops.p_base + 1, n_out = restrict ? ops.p_base + 1 : ops.p + 1;
            const int dim = ShapeDim(ops.shape);
            const int need = (dim == 3) ? std::max(n_in * n_in * n_out, n_in * n_out * n_out) : n_in * n_out;
            const unsigned grid = GridFor(ncells);
            if (dim == 3) {
                if (need <= 64) hipLaunchKernelGGL((Tensor3Kernel<64>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
                else if (need <= 128) hipLaunchKernelGGL((Tensor3Kernel<128>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
                else if (need <= 256) hipLaunchKernelGGL((Tensor3Kernel<256>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
                else if (need <= 512) hipLaunchKernelGGL((Tensor3Kernel<512>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
                else throw std::runtime_error("device tensor apply: order too high for the register-resident sweep");
            } else if (dim == 2) {
                if (need <= 64) hipLaunchKernelGGL((Tensor2Kernel<64>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
                else hipLaunchKernelGGL((Tensor2Kernel<256>), dim3(grid), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
            } else {
                hipLaunchKernelGGL(DenseApplyKernel, dim3(GridFor(ncells * nsp_out)), dim3(kThreads), 0, 0, M, nsp_out, nsp_in, (const double*)d_in, (double*)d_out, ncells);
            }
        }
        HPM_DEV_CHECK(hipGetLastError());
    }

    // ---- resident component ---------------------------------------------------------------------------
    void BeginComponent(const double* u, size_t npts) override {
        d_u_.reserve(npts * 8);
        HPM_DEV_CHECK(hipMemcpy(d_u_.p, u, npts * 8, hipMemcpyHostToDevice));
        resident_ = u; resident_n_ = npts;
    }
    void EndComponent() override { resident_ = nullptr; resident_n_ = 0; d_k_n_ = 0; }
    bool KeepsDeviceSymbols() const override { return true; }
    const int32_t* DeviceSymbols() const override { return d_k_n_ ? static_cast<const int32_t*>(d_k_.p) : nullptr; }
    void CopySymbolsToHost(int32_t* k) const override { if (d_k_n_) HPM_DEV_CHECK(hipMemcpy(k, d_k_.p, d_k_n_ * 4, hipMemcpyDeviceToHost)); }
    void* HostScratch(int slot, size_t bytes) override { pinned_[slot].reserve(bytes); return pinned_[slot].p; }

    void Restrict(const OperatorSet& ops, const double* u, size_t ncells, double* base) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        d_out_.reserve(nbase * 8);
        Apply(ops, true, d_u_.p, ncells, d_out_.p);
        HPM_DEV_CHECK(hipMemcpy(base, d_out_.p, nbase * 8, hipMemcpyDeviceToHost));
    }
    void Prolong(const OperatorSet& ops, const double* base, size_t ncells, double* u) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        d_base_.reserve(nbase * 8); HPM_DEV_CHECK(hipMemcpy(d_base_.p, base, nbase * 8, hipMemcpyHostToDevice));
        d_out_.reserve(npts * 8);
        Apply(ops, false, d_base_.p, ncells, d_out_.p);
        HPM_DEV_CHECK(hipMemcpy(u, d_out_.p, npts * 8, hipMemcpyDeviceToHost));
    }
    bool ResidualFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, double q, int32_t* k, const double* u) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        d_base_.reserve(nbase * 8); HPM_DEV_CHECK(hipMemcpy(d_base_.p, base_dec, nbase * 8, hipMemcpyHostToDevice));
        d_out_.reserve(npts * 8);
        Apply(ops, false, d_base_.p, ncells, d_out_.p);                 // d_out_ = P base_dec (stays on the device)
        d_k_.reserve(npts * 4); d_flag_.reserve(4); HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        hipLaunchKernelGGL(QuantizeKernel, dim3(GridFor(npts)), dim3(kThreads), 0, 0, (const double*)d_u_.p, (const double*)d_out_.p, npts, 1.0 / q, (int32_t*)d_k_.p, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        int overflow = 0; HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        d_k_n_ = npts;
        if (k) CopySymbolsToHost(k);
        return overflow == 0;
    }
    void ReconstructFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, const int32_t* k, double q, double* out) override {
        const size_t npts = ncells * (size_t)ops.nsp_data;
        d_k_.reserve(npts * 4); HPM_DEV_CHECK(hipMemcpy(d_k_.p, k, npts * 4, hipMemcpyHostToDevice)); d_k_n_ = 0;
        ReconstructDevice(ops, base_dec, ncells, static_cast<const int32_t*>(d_k_.p), q, out);
    }
    bool ReconstructFromDeviceSymbols(const OperatorSet& ops, const double* base_dec, size_t ncells, const int32_t* d_k, double q, double* out) override {
        ReconstructDevice(ops, base_dec, ncells, d_k, q, out);
        return true;
    }
    void ReconstructDevice(const OperatorSet& ops, const double* base_dec, size_t ncells, const int32_t* d_k, double q, double* out) {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        d_base_.reserve(nbase * 8); HPM_DEV_CHECK(hipMemcpy(d_base_.p, base_dec, nbase * 8, hipMemcpyHostToDevice));
        d_out_.reserve(npts * 8);
        Apply(ops, false, d_base_.p, ncells, d_out_.p);
        hipLaunchKernelGGL(DequantizeAddKernel, dim3(GridFor(npts)), dim3(kThreads), 0, 0, d_k, npts, q, (double*)d_out_.p);
        HPM_DEV_CHECK(hipGetLastError());
        HPM_DEV_CHECK(hipMemcpy(out, d_out_.p, npts * 8, hipMemcpyDeviceToHost));
    }
    bool QuantizeImpl(const double* x, const double* sub, size_t n, double q, int32_t* k) {
        d_in_.reserve(n * 8); d_k_.reserve(n * 4); d_flag_.reserve(4);
        HPM_DEV_CHECK(hipMemcpy(d_in_.p, x, n * 8, hipMemcpyHostToDevice));
        if (sub) { d_out_.reserve(n * 8); HPM_DEV_CHECK(hipMemcpy(d_out_.p, sub, n * 8, hipMemcpyHostToDevice)); }
        HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        hipLaunchKernelGGL(QuantizeKernel, dim3(GridFor(n)), dim3(kThreads), 0, 0, (const double*)d_in_.p, sub ? (const double*)d_out_.p : nullptr, n, 1.0 / q, (int32_t*)d_k_.p, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        int overflow = 0;
        HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        HPM_DEV_CHECK(hipMemcpy(k, d_k_.p, n * 4, hipMemcpyDeviceToHost));
        d_k_n_ = n;
        return overflow == 0;
    }
    bool Quantize(const double* x, size_t n, double q, int32_t* k) override { return QuantizeImpl(x, nullptr, n, q, k); }
    bool ResidualQuantize(const double* u, const double* ub, size_t n, double q, int32_t* k) override { return QuantizeImpl(u, ub, n, q, k); }
    void DequantizeAdd(const int32_t* k, size_t n, double q, double* u) override {
        d_k_.reserve(n * 4); d_in_.reserve(n * 8);
        HPM_DEV_CHECK(hipMemcpy(d_k_.p, k, n * 4, hipMemcpyHostToDevice)); HPM_DEV_CHECK(hipMemcpy(d_in_.p, u, n * 8, hipMemcpyHostToDevice)); d_k_n_ = 0;
        hipLaunchKernelGGL(DequantizeAddKernel, dim3(GridFor(n)), dim3(kThreads), 0, 0, (const int32_t*)d_k_.p, n, q, (double*)d_in_.p);
        HPM_DEV_CHECK(hipGetLastError());
        HPM_DEV_CHECK(hipMemcpy(u, d_in_.p, n * 8, hipMemcpyDeviceToHost));
    }
    void CellMeans(const double* base, size_t ncells, int nb, double* avg) override {
        d_in_.reserve(ncells * nb * 8); d_out_.reserve(ncells * 8);
        HPM_DEV_CHECK(hipMemcpy(d_in_.p, base, ncells * nb * 8, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(CellMeansKernel, dim3(GridFor(ncells)), dim3(kThreads), 0, 0, (const double*)d_in_.p, ncells, nb, (double*)d_out_.p);
        HPM_DEV_CHECK(hipGetLastError());
        HPM_DEV_CHECK(hipMemcpy(avg, d_out_.p, ncells * 8, hipMemcpyDeviceToHost));
    }
private:
    DeviceBuffer d_in_, d_out_, d_m_, d_k_, d_flag_, d_u_, d_base_;
    PinnedBuffer pinned_[NumHostSlots];
    std::vector<double> op_cache_;
    const double* resident_ = nullptr; size_t resident_n_ = 0;
    size_t d_k_n_ = 0;   // symbols currently valid in d_k_ (see DeviceSymbols)
};

// The process-wide instance (device memory, pinned scratch and the uploaded operator persist); callers get
// a non-owning proxy so MakeBackend's unique_ptr semantics stay unchanged.
DeviceBackend* Instance() {
    static DeviceBackend* inst = nullptr; static bool tried = false;
    if (!tried) { tried = true; try { inst = new DeviceBackend(0); } catch (...) { inst = nullptr; } }
    return inst;
}

class Proxy : public Backend {
public:
    explicit Proxy(DeviceBackend& b) : b_(b) {}
    DeviceType type() const override { return b_.type(); }
    const char* name() const override { return b_.name(); }
    void Restrict(const OperatorSet& o, const double* u, size_t n, double* b) override { b_.Restrict(o, u, n, b); }
    void Prolong(const OperatorSet& o, const double* b, size_t n, double* u) override { b_.Prolong(o, b, n, u); }
    bool Quantize(const double* x, size_t n, double q, int32_t* k) override { return b_.Quantize(x, n, q, k); }
    bool ResidualQuantize(const double* u, const double* ub, size_t n, double q, int32_t* k) override { return b_.ResidualQuantize(u, ub, n, q, k); }
    void DequantizeAdd(const int32_t* k, size_t n, double q, double* u) override { b_.DequantizeAdd(k, n, q, u); }
    void CellMeans(const double* b, size_t n, int nb, double* a) override { b_.CellMeans(b, n, nb, a); }
    void BeginComponent(const double* u, size_t n) override { b_.BeginComponent(u, n); }
    void EndComponent() override { b_.EndComponent(); }
    bool KeepsDeviceSymbols() const override { return true; }
    const int32_t* DeviceSymbols() const override { return b_.DeviceSymbols(); }
    void CopySymbolsToHost(int32_t* k) const override { b_.CopySymbolsToHost(k); }
    bool ResidualFromBase(const OperatorSet& o, const double* bd, size_t n, double q, int32_t* k, const double* u) override { return b_.ResidualFromBase(o, bd, n, q, k, u); }
    void ReconstructFromBase(const OperatorSet& o, const double* bd, size_t n, const int32_t* k, double q, double* out) override { b_.ReconstructFromBase(o, bd, n, k, q, out); }
    bool ReconstructFromDeviceSymbols(const OperatorSet& o, const double* bd, size_t n, const int32_t* dk, double q, double* out) override { return b_.ReconstructFromDeviceSymbols(o, bd, n, dk, q, out); }
    void* HostScratch(int slot, size_t bytes) override { return b_.HostScratch(slot, bytes); }
private:
    DeviceBackend& b_;
};

} // namespace

std::unique_ptr<Backend> HPM_MAKE_BACKEND_FN() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return nullptr;
    DeviceBackend* inst = Instance();
    if (!inst) return nullptr;
    return std::unique_ptr<Backend>(new Proxy(*inst));
}

} // namespace hpmgard
