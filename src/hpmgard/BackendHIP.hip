// HIP backend, first version: dense per-cell restriction/prolongation (the tensor-product operator is
// expanded to its full nsp_base x nsp_data matrix on the host, so HEX and simplex shapes share one
// kernel) and elementwise quantisation kernels. Host arrays are staged through device buffers per call;
// keeping data resident across the stages of a component is the next optimisation.
#include "hpmgard/Backend.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpmgard {

namespace {

#define HPM_HIP_CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e) + " at " #call); } while (0)

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

// Full per-cell matrix of a tensor OperatorSet (Kronecker of the 1-D matrix along each axis, i fastest).
std::vector<double> FullMatrix(const std::vector<double>& M1, int n_out, int n_in, int dim) {
    std::vector<double> full(1, 1.0); int ro = 1, ci = 1;
    for (int d = 0; d < dim; ++d) {
        // new index = old_index + old_size * new_axis_index  (new axis slower than the existing ones)
        std::vector<double> next((size_t)ro * n_out * ci * n_in);
        for (int a = 0; a < n_out; ++a) for (int i = 0; i < n_in; ++i)
            for (int r = 0; r < ro; ++r) for (int c = 0; c < ci; ++c)
                next[((size_t)a * ro + r) * ((size_t)ci * n_in) + ((size_t)i * ci + c)] = M1[(size_t)a * n_in + i] * full[(size_t)r * ci + c];
        full.swap(next); ro *= n_out; ci *= n_in;
    }
    return full;
}

struct DeviceBuffer {
    void* p = nullptr; size_t bytes = 0;
    void reserve(size_t b) { if (b > bytes) { if (p) (void)hipFree(p); HPM_HIP_CHECK(hipMalloc(&p, b)); bytes = b; } }
    ~DeviceBuffer() { if (p) (void)hipFree(p); }
};

class HipBackend : public Backend {
public:
    HipBackend(int device) { HPM_HIP_CHECK(hipSetDevice(device)); }
    DeviceType type() const override { return DeviceType::HIP; }
    const char* name() const override { return "hip"; }

    void Apply(const std::vector<double>& full, int n_out, int n_in, const double* in, size_t ncells, double* out) {
        const size_t nin = ncells * n_in, nout = ncells * n_out;
        d_in_.reserve(nin * 8); d_out_.reserve(nout * 8); d_m_.reserve(full.size() * 8);
        HPM_HIP_CHECK(hipMemcpy(d_m_.p, full.data(), full.size() * 8, hipMemcpyHostToDevice));
        HPM_HIP_CHECK(hipMemcpy(d_in_.p, in, nin * 8, hipMemcpyHostToDevice));
        const int threads = 256; const size_t blocks = (nout + threads - 1) / threads;
        hipLaunchKernelGGL(DenseApplyKernel, dim3(blocks), dim3(threads), 0, 0, (const double*)d_m_.p, n_out, n_in, (const double*)d_in_.p, (double*)d_out_.p, ncells);
        HPM_HIP_CHECK(hipGetLastError());
        HPM_HIP_CHECK(hipMemcpy(out, d_out_.p, nout * 8, hipMemcpyDeviceToHost));
    }
    // resident component
    void BeginComponent(const double* u, size_t npts) override {
        d_u_.reserve(npts * 8);
        HPM_HIP_CHECK(hipMemcpy(d_u_.p, u, npts * 8, hipMemcpyHostToDevice));
        resident_ = u; resident_n_ = npts;
    }
    void EndComponent() override { resident_ = nullptr; resident_n_ = 0; d_k_n_ = 0; }
    const int32_t* DeviceSymbols() const override { return d_k_n_ ? static_cast<const int32_t*>(d_k_.p) : nullptr; }
    void UploadMatrix(const std::vector<double>& full) { d_m_.reserve(full.size() * 8); HPM_HIP_CHECK(hipMemcpy(d_m_.p, full.data(), full.size() * 8, hipMemcpyHostToDevice)); }
    void Launch(const void* d_in, int n_out, int n_in, size_t ncells, void* d_out) {
        const size_t nout = ncells * n_out; const int threads = 256; const size_t blocks = (nout + threads - 1) / threads;
        hipLaunchKernelGGL(DenseApplyKernel, dim3(blocks), dim3(threads), 0, 0, (const double*)d_m_.p, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
        HPM_HIP_CHECK(hipGetLastError());
    }
    void Restrict(const OperatorSet& ops, const double* u, size_t ncells, double* base) override {
        const std::vector<double> full = ops.tensor ? FullMatrix(ops.RE1, ops.p_base + 1, ops.p + 1, ShapeDim(ops.shape)) : ops.Rfull;
        if (u == resident_ && resident_n_ == ncells * (size_t)ops.nsp_data) {   // reuse the device copy
            UploadMatrix(full); d_out_.reserve(ncells * ops.nsp_base * 8);
            Launch(d_u_.p, ops.nsp_base, ops.nsp_data, ncells, d_out_.p);
            HPM_HIP_CHECK(hipMemcpy(base, d_out_.p, ncells * ops.nsp_base * 8, hipMemcpyDeviceToHost));
            return;
        }
        Apply(full, ops.nsp_base, ops.nsp_data, u, ncells, base);
    }
    bool ResidualFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, double q, int32_t* k, const double* u) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        const std::vector<double> full = ops.tensor ? FullMatrix(ops.Pc1, ops.p + 1, ops.p_base + 1, ShapeDim(ops.shape)) : ops.Pfull;
        UploadMatrix(full);
        d_base_.reserve(nbase * 8); HPM_HIP_CHECK(hipMemcpy(d_base_.p, base_dec, nbase * 8, hipMemcpyHostToDevice));
        d_out_.reserve(npts * 8);
        Launch(d_base_.p, ops.nsp_data, ops.nsp_base, ncells, d_out_.p);          // d_out_ = P base_dec (stays on the device)
        d_k_.reserve(npts * 4); d_flag_.reserve(4); HPM_HIP_CHECK(hipMemset(d_flag_.p, 0, 4));
        const int threads = 256; const size_t blocks = (npts + threads - 1) / threads;
        hipLaunchKernelGGL(QuantizeKernel, dim3(blocks), dim3(threads), 0, 0, (const double*)d_u_.p, (const double*)d_out_.p, npts, 1.0 / q, (int32_t*)d_k_.p, (int*)d_flag_.p);
        HPM_HIP_CHECK(hipGetLastError());
        int overflow = 0; HPM_HIP_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        HPM_HIP_CHECK(hipMemcpy(k, d_k_.p, npts * 4, hipMemcpyDeviceToHost));
        d_k_n_ = npts;
        return overflow == 0;
    }
    void ReconstructFromBase(const OperatorSet& ops, const double* base_dec, size_t ncells, const int32_t* k, double q, double* out) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        const std::vector<double> full = ops.tensor ? FullMatrix(ops.Pc1, ops.p + 1, ops.p_base + 1, ShapeDim(ops.shape)) : ops.Pfull;
        UploadMatrix(full);
        d_base_.reserve(nbase * 8); HPM_HIP_CHECK(hipMemcpy(d_base_.p, base_dec, nbase * 8, hipMemcpyHostToDevice));
        d_out_.reserve(npts * 8);
        Launch(d_base_.p, ops.nsp_data, ops.nsp_base, ncells, d_out_.p);
        d_k_.reserve(npts * 4); HPM_HIP_CHECK(hipMemcpy(d_k_.p, k, npts * 4, hipMemcpyHostToDevice)); d_k_n_ = 0;
        const int threads = 256; const size_t blocks = (npts + threads - 1) / threads;
        hipLaunchKernelGGL(DequantizeAddKernel, dim3(blocks), dim3(threads), 0, 0, (const int32_t*)d_k_.p, npts, q, (double*)d_out_.p);
        HPM_HIP_CHECK(hipGetLastError());
        HPM_HIP_CHECK(hipMemcpy(out, d_out_.p, npts * 8, hipMemcpyDeviceToHost));
    }
    void Prolong(const OperatorSet& ops, const double* base, size_t ncells, double* u) override {
        const std::vector<double> full = ops.tensor ? FullMatrix(ops.Pc1, ops.p + 1, ops.p_base + 1, ShapeDim(ops.shape)) : ops.Pfull;
        Apply(full, ops.nsp_data, ops.nsp_base, base, ncells, u);
    }
    bool QuantizeImpl(const double* x, const double* sub, size_t n, double q, int32_t* k) {
        d_in_.reserve(n * 8); d_k_.reserve(n * 4); d_flag_.reserve(4);
        HPM_HIP_CHECK(hipMemcpy(d_in_.p, x, n * 8, hipMemcpyHostToDevice));
        if (sub) { d_out_.reserve(n * 8); HPM_HIP_CHECK(hipMemcpy(d_out_.p, sub, n * 8, hipMemcpyHostToDevice)); }
        HPM_HIP_CHECK(hipMemset(d_flag_.p, 0, 4));
        const int threads = 256; const size_t blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(QuantizeKernel, dim3(blocks), dim3(threads), 0, 0, (const double*)d_in_.p, sub ? (const double*)d_out_.p : nullptr, n, 1.0 / q, (int32_t*)d_k_.p, (int*)d_flag_.p);
        HPM_HIP_CHECK(hipGetLastError());
        int overflow = 0;
        HPM_HIP_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        HPM_HIP_CHECK(hipMemcpy(k, d_k_.p, n * 4, hipMemcpyDeviceToHost));
        d_k_n_ = n;
        return overflow == 0;
    }
    bool Quantize(const double* x, size_t n, double q, int32_t* k) override { return QuantizeImpl(x, nullptr, n, q, k); }
    bool ResidualQuantize(const double* u, const double* ub, size_t n, double q, int32_t* k) override { return QuantizeImpl(u, ub, n, q, k); }
    void DequantizeAdd(const int32_t* k, size_t n, double q, double* u) override {
        d_k_.reserve(n * 4); d_in_.reserve(n * 8);
        HPM_HIP_CHECK(hipMemcpy(d_k_.p, k, n * 4, hipMemcpyHostToDevice)); HPM_HIP_CHECK(hipMemcpy(d_in_.p, u, n * 8, hipMemcpyHostToDevice)); d_k_n_ = 0;
        const int threads = 256; const size_t blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(DequantizeAddKernel, dim3(blocks), dim3(threads), 0, 0, (const int32_t*)d_k_.p, n, q, (double*)d_in_.p);
        HPM_HIP_CHECK(hipGetLastError());
        HPM_HIP_CHECK(hipMemcpy(u, d_in_.p, n * 8, hipMemcpyDeviceToHost));
    }
    void CellMeans(const double* base, size_t ncells, int nb, double* avg) override {
        d_in_.reserve(ncells * nb * 8); d_out_.reserve(ncells * 8);
        HPM_HIP_CHECK(hipMemcpy(d_in_.p, base, ncells * nb * 8, hipMemcpyHostToDevice));
        const int threads = 256; const size_t blocks = (ncells + threads - 1) / threads;
        hipLaunchKernelGGL(CellMeansKernel, dim3(blocks), dim3(threads), 0, 0, (const double*)d_in_.p, ncells, nb, (double*)d_out_.p);
        HPM_HIP_CHECK(hipGetLastError());
        HPM_HIP_CHECK(hipMemcpy(avg, d_out_.p, ncells * 8, hipMemcpyDeviceToHost));
    }
private:
    DeviceBuffer d_in_, d_out_, d_m_, d_k_, d_flag_, d_u_, d_base_;
    const double* resident_ = nullptr; size_t resident_n_ = 0;
    size_t d_k_n_ = 0;   // symbols currently valid in d_k_ (see DeviceSymbols)
};

} // namespace

std::unique_ptr<Backend> MakeHipBackend() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return nullptr;
    try { return std::unique_ptr<Backend>(new HipBackend(0)); } catch (...) { return nullptr; }
}

} // namespace hpmgard
