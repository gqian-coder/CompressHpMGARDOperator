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

// Prolongation (few inputs per cell, many outputs): one thread per output value, Kronecker weights
// evaluated on the fly in the CPU sweep's summation order (i, then j, then k). n_in is small (p_base + 1).
__global__ void Tensor3ProlongKernel(const double* __restrict__ M, int n_out, int n_in, const double* __restrict__ in, double* __restrict__ out, size_t ncells) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int nout3 = n_out * n_out * n_out, nin3 = n_in * n_in * n_in;
    const size_t total = ncells * (size_t)nout3;
    if (idx >= total) return;
    const size_t c = idx / nout3; const int r = static_cast<int>(idx - c * nout3);
    const int a = r % n_out, b = (r / n_out) % n_out, cc = r / (n_out * n_out);
    const double* x = in + c * (size_t)nin3;
    const double* ma = M + a * n_in; const double* mb = M + b * n_in; const double* mc = M + cc * n_in;
    double s = 0.0;
    for (int k = 0; k < n_in; ++k) {
        double sk = 0.0;
        for (int j = 0; j < n_in; ++j) {
            double sj = 0.0; const double* xs = x + (k * n_in + j) * n_in;
            for (int i = 0; i < n_in; ++i) sj += ma[i] * xs[i];
            sk += mb[j] * sj;
        }
        s += mc[k] * sk;
    }
    out[idx] = s;
}
__global__ void Tensor2ProlongKernel(const double* __restrict__ M, int n_out, int n_in, const double* __restrict__ in, double* __restrict__ out, size_t ncells) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int nout2 = n_out * n_out, nin2 = n_in * n_in;
    const size_t total = ncells * (size_t)nout2;
    if (idx >= total) return;
    const size_t c = idx / nout2; const int r = static_cast<int>(idx - c * nout2);
    const int a = r % n_out, b = r / n_out;
    const double* x = in + c * (size_t)nin2;
    const double* ma = M + a * n_in; const double* mb = M + b * n_in;
    double s = 0.0;
    for (int j = 0; j < n_in; ++j) {
        double sj = 0.0; const double* xs = x + j * n_in;
        for (int i = 0; i < n_in; ++i) sj += ma[i] * xs[i];
        s += mb[j] * sj;
    }
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

// base split kernels (device-resident base path)
__global__ void DequantKernel(const int32_t* __restrict__ k, size_t n, double q, double* __restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = k[i] * q;
}
// cell average split: k_rem = rint((base - avg_dec[cell]) / qb), base_dec = avg_dec[cell] + k_rem * qb
__global__ void CellAvgRemainderKernel(const double* __restrict__ base, const double* __restrict__ avg_dec, size_t nbase, int nb, double qb, double inv_qb,
                                       int32_t* __restrict__ k, double* __restrict__ base_dec, int* __restrict__ overflow) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nbase) return;
    const double a = avg_dec[i / nb];
    const double v = (base[i] - a) * inv_qb;
    if (!(fabs(v) < kMaxQuantMagnitude)) { atomicOr(overflow, 1); k[i] = 0; base_dec[i] = a; return; }
    const int32_t kk = static_cast<int32_t>(llrint(v));
    k[i] = kk; base_dec[i] = a + kk * qb;
}
// dedup split: per-vertex sums (atomic; summation order is not deterministic, so the quantised means may
// differ in rounding ties between runs/backends: both streams are valid), then remainders in Hilbert order
__global__ void GroupSumKernel(const double* __restrict__ base, const uint32_t* __restrict__ vid, size_t nbase, double* __restrict__ sum) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nbase) atomicAdd(sum + vid[i], base[i]);
}
__global__ void GroupMeanKernel(const double* __restrict__ sum, const uint32_t* __restrict__ cnt, size_t ngroups, double* __restrict__ avg) {
    const size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (g < ngroups) avg[g] = cnt[g] ? sum[g] / cnt[g] : 0.0;
}
// slot -> d = perm[slot]; k_rem[slot] = rint((base[d] - avg_dec[vid[d]]) / qb); base_dec[d] = avg_dec[vid[d]] + k_rem[slot] * qb
__global__ void DedupRemainderKernel(const double* __restrict__ base, const double* __restrict__ avg_dec, const uint32_t* __restrict__ vid, const uint32_t* __restrict__ perm,
                                     size_t nbase, double qb, double inv_qb, int32_t* __restrict__ k, double* __restrict__ base_dec, int* __restrict__ overflow) {
    const size_t slot = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (slot >= nbase) return;
    const uint32_t d = perm[slot];
    const double a = avg_dec[vid[d]];
    const double v = (base[d] - a) * inv_qb;
    if (!(fabs(v) < kMaxQuantMagnitude)) { atomicOr(overflow, 1); k[slot] = 0; base_dec[d] = a; return; }
    const int32_t kk = static_cast<int32_t>(llrint(v));
    k[slot] = kk; base_dec[d] = a + kk * qb;
}

// per-block partial statistics (min, max, sum of squares) of a component
__global__ void StatsKernel(const double* __restrict__ u, size_t n, double* __restrict__ partial) {
    __shared__ double smn[256], smx[256], sss[256];
    double lo = 1e300, hi = -1e300, ss = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const double v = u[i]; lo = fmin(lo, v); hi = fmax(hi, v); ss += v * v; }
    smn[threadIdx.x] = lo; smx[threadIdx.x] = hi; sss[threadIdx.x] = ss;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smn[threadIdx.x] = fmin(smn[threadIdx.x], smn[threadIdx.x + s]); smx[threadIdx.x] = fmax(smx[threadIdx.x], smx[threadIdx.x + s]); sss[threadIdx.x] += sss[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { partial[3 * blockIdx.x] = smn[0]; partial[3 * blockIdx.x + 1] = smx[0]; partial[3 * blockIdx.x + 2] = sss[0]; }
}

// Fused per-cell pipeline. One thread per data node, cells_per_block cells per thread block, dense R
// (nb x nsp) and P (nsp x nb) from global memory, element values in shared memory.
// mode 0 plain: k_base per corner -> d_rem (n = nbase), no avg stream
// mode 1 cellavg: k_avg per cell -> d_avg, k_rem per corner -> d_rem
// mode 2 dedup: avg_dec per group given (gavg), k_rem per Hilbert slot (inv_perm) -> d_rem
// mode 3 dedup sums: restrict only and atomically add the corner values into gsum[vid[d]] (no symbols)
__global__ void FusedCellKernel(const double* __restrict__ u, size_t ncells, int nsp, int nb, int cells_per_block,
                                const double* __restrict__ R, const double* __restrict__ P, int mode,
                                const double* __restrict__ gavg, const uint32_t* __restrict__ vid, const uint32_t* __restrict__ inv_perm, double* __restrict__ gsum,
                                double qb, double inv_qb, double qres, double inv_qres,
                                int32_t* __restrict__ k_avg, int32_t* __restrict__ k_rem, int32_t* __restrict__ k_res, int* __restrict__ overflow) {
    extern __shared__ double sm[];
    const int cl = threadIdx.x / nsp, node = threadIdx.x - cl * nsp;
    const size_t cell = blockIdx.x * (size_t)cells_per_block + cl;
    const bool active = cl < cells_per_block && cell < ncells;
    double* x = sm + (size_t)cl * (nsp + 2 * nb + 1); double* base = x + nsp; double* bdec = base + nb; double* avg = bdec + nb;
    if (active) x[node] = u[cell * (size_t)nsp + node];
    __syncthreads();
    if (active && node < nb) { double sacc = 0.0; const double* r = R + (size_t)node * nsp; for (int i = 0; i < nsp; ++i) sacc += r[i] * x[i]; base[node] = sacc; }
    __syncthreads();
    if (mode == 3) { if (active && node < nb) atomicAdd(gsum + vid[cell * (size_t)nb + node], base[node]); return; }
    if (mode == 1) {
        if (active && node == 0) {
            double sacc = 0.0; for (int j = 0; j < nb; ++j) sacc += base[j];
            const double a = sacc / nb, v = a * inv_qb;
            int32_t k = 0; if (!(fabs(v) < kMaxQuantMagnitude)) atomicOr(overflow, 1); else k = static_cast<int32_t>(llrint(v));
            k_avg[cell] = k; avg[0] = k * qb;
        }
        __syncthreads();
    }
    if (active && node < nb) {
        const size_t d = cell * (size_t)nb + node;
        double a = 0.0;
        if (mode == 1) a = avg[0]; else if (mode == 2) a = gavg[vid[d]];
        int32_t k = 0;
        if (!(mode == 1 && nb == 1)) {   // a one-value cell base has no remainder stream (the host path omits it too)
            const double v = (base[node] - a) * inv_qb;
            if (!(fabs(v) < kMaxQuantMagnitude)) atomicOr(overflow, 1); else k = static_cast<int32_t>(llrint(v));
            k_rem[mode == 2 ? inv_perm[d] : d] = k;
        }
        bdec[node] = a + k * qb;
    }
    __syncthreads();
    if (active) {
        double sacc = 0.0; const double* pr = P + (size_t)node * nb; for (int j = 0; j < nb; ++j) sacc += pr[j] * bdec[j];
        const double v = (x[node] - sacc) * inv_qres;
        int32_t k = 0; if (!(fabs(v) < kMaxQuantMagnitude)) atomicOr(overflow, 1); else k = static_cast<int32_t>(llrint(v));
        k_res[cell * (size_t)nsp + node] = k;
    }
}
// Fused reconstruction: out = P (decoded base) + k_res * qres. mode as above (0 plain: k per corner in k_rem; 1 cellavg: k_avg per cell; 2 dedup: gavg per group, k_rem per slot)
__global__ void ReconstructFusedKernel(size_t ncells, int nsp, int nb, int cells_per_block, const double* __restrict__ P, int mode,
                                       const int32_t* __restrict__ k_avg, const int32_t* __restrict__ k_rem, const int32_t* __restrict__ k_res,
                                       const double* __restrict__ gavg, const uint32_t* __restrict__ vid, const uint32_t* __restrict__ inv_perm,
                                       double qb, double qres, double* __restrict__ out) {
    extern __shared__ double sm[];
    const int cl = threadIdx.x / nsp, node = threadIdx.x - cl * nsp;
    const size_t cell = blockIdx.x * (size_t)cells_per_block + cl;
    const bool active = cl < cells_per_block && cell < ncells;
    double* bdec = sm + (size_t)cl * nb;
    if (active && node < nb) {
        const size_t d = cell * (size_t)nb + node;
        double a = 0.0;
        if (mode == 1) a = k_avg[cell] * qb; else if (mode == 2) a = gavg[vid[d]];
        const int32_t k = k_rem ? k_rem[mode == 2 ? inv_perm[d] : d] : 0;
        bdec[node] = a + k * qb;
    }
    __syncthreads();
    if (active) {
        double sacc = 0.0; const double* pr = P + (size_t)node * nb; for (int j = 0; j < nb; ++j) sacc += pr[j] * bdec[j];
        out[cell * (size_t)nsp + node] = sacc + k_res[cell * (size_t)nsp + node] * qres;
    }
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
            if (!restrict && dim == 3) {          // prolongation: one thread per output value
                hipLaunchKernelGGL(Tensor3ProlongKernel, dim3(GridFor(ncells * (size_t)nsp_out)), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
            } else if (!restrict && dim == 2) {
                hipLaunchKernelGGL(Tensor2ProlongKernel, dim3(GridFor(ncells * (size_t)nsp_out)), dim3(kThreads), 0, 0, M, n_out, n_in, (const double*)d_in, (double*)d_out, ncells);
            } else if (dim == 3) {
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
    void EndComponent() override { resident_ = nullptr; resident_n_ = 0; d_k_n_ = 0; d_basedec_valid_ = false; }
    bool KeepsDeviceSymbols() const override { return true; }
    const int32_t* DeviceSymbols() const override { return d_k_n_ ? static_cast<const int32_t*>(d_k_.p) : nullptr; }
    void CopySymbolsToHost(int32_t* k) const override { if (d_k_n_) HPM_DEV_CHECK(hipMemcpy(k, d_k_.p, d_k_n_ * 4, hipMemcpyDeviceToHost)); }
    void* HostScratch(int slot, size_t bytes) override { pinned_[slot].reserve(bytes); return pinned_[slot].p; }

    // ---- statistics on the device -----------------------------------------------------------------------
    void ComponentStats(const double* u, size_t npts, double& lo, double& hi, double& sumsq) override {
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        const unsigned grid = static_cast<unsigned>(std::min<size_t>(1024, (npts + 255) / 256));
        d_partial_.reserve(grid * 3 * 8);
        hipLaunchKernelGGL(StatsKernel, dim3(grid), dim3(256), 0, 0, (const double*)d_u_.p, npts, (double*)d_partial_.p);
        HPM_DEV_CHECK(hipGetLastError());
        std::vector<double> part(grid * 3);
        HPM_DEV_CHECK(hipMemcpy(part.data(), d_partial_.p, grid * 3 * 8, hipMemcpyDeviceToHost));
        lo = 1e300; hi = -1e300; sumsq = 0.0;
        for (unsigned b = 0; b < grid; ++b) { lo = std::min(lo, part[3 * b]); hi = std::max(hi, part[3 * b + 1]); sumsq += part[3 * b + 2]; }
    }

    // ---- fused path ------------------------------------------------------------------------------------
    int CellsPerBlock(int nsp, int nb) const {
        if (nsp > 1024) return 0;
        const size_t per_cell = static_cast<size_t>(nsp + 2 * nb + 1) * 8;
        int cpb = std::max(1, std::min(1024 / nsp, static_cast<int>((48u * 1024u) / per_cell)));
        return cpb;
    }
    bool SupportsFused(const OperatorSet& ops) const override { return CellsPerBlock(ops.nsp_data, ops.nsp_base) > 0 && !ops.Rfull.empty() && !ops.Pfull.empty(); }
    void UploadDense(const OperatorSet& ops) {
        if (dense_key_ == ops.key && !dense_key_.empty()) return;
        d_R_.reserve(ops.Rfull.size() * 8); d_P_.reserve(ops.Pfull.size() * 8);
        HPM_DEV_CHECK(hipMemcpy(d_R_.p, ops.Rfull.data(), ops.Rfull.size() * 8, hipMemcpyHostToDevice));
        HPM_DEV_CHECK(hipMemcpy(d_P_.p, ops.Pfull.data(), ops.Pfull.size() * 8, hipMemcpyHostToDevice));
        dense_key_ = ops.key;
    }
    void UploadGeometry(const DedupGeometry& geom, size_t nbase) {
        if (geom_key_ == &geom && geom_n_ == nbase) return;
        const size_t ng = geom.ngroups;
        std::vector<uint32_t> cnt(ng, 0), inv(nbase, 0);
        for (uint32_t v : geom.vid) ++cnt[v];
        for (size_t slot = 0; slot < nbase; ++slot) inv[geom.perm[slot]] = static_cast<uint32_t>(slot);
        d_vid_.reserve(nbase * 4); d_perm_.reserve(nbase * 4); d_inv_.reserve(nbase * 4); d_cnt_.reserve(ng * 4);
        HPM_DEV_CHECK(hipMemcpy(d_vid_.p, geom.vid.data(), nbase * 4, hipMemcpyHostToDevice));
        HPM_DEV_CHECK(hipMemcpy(d_perm_.p, geom.perm.data(), nbase * 4, hipMemcpyHostToDevice));
        HPM_DEV_CHECK(hipMemcpy(d_inv_.p, inv.data(), nbase * 4, hipMemcpyHostToDevice));
        HPM_DEV_CHECK(hipMemcpy(d_cnt_.p, cnt.data(), ng * 4, hipMemcpyHostToDevice));
        geom_key_ = &geom; geom_n_ = nbase;
    }
    bool FusedCompress(const OperatorSet& ops, size_t ncells, BaseMethod method, const DedupGeometry* geom, double qb, double qres, const double* u, FusedStreams& out) override {
        const int nsp = ops.nsp_data, nb = ops.nsp_base; const size_t npts = ncells * (size_t)nsp, nbase = ncells * (size_t)nb;
        const int cpb = CellsPerBlock(nsp, nb); if (cpb <= 0) return false;
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        UploadDense(ops);
        const int mode = method == BaseMethod::Plain ? 0 : method == BaseMethod::CellAvg ? 1 : 2;
        if (mode == 2) { if (!geom || geom->vid.size() != nbase) return false; UploadGeometry(*geom, nbase); }
        const size_t ng = mode == 2 ? geom->ngroups : 0;
        d_kavg_.reserve(std::max<size_t>(mode == 2 ? ng : ncells, 1) * 4); d_krem_.reserve(nbase * 4); d_k_.reserve(npts * 4); d_flag_.reserve(4);
        HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        const unsigned grid = static_cast<unsigned>((ncells + cpb - 1) / cpb); const unsigned threads = static_cast<unsigned>(cpb * nsp);
        const size_t shm = static_cast<size_t>(cpb) * (nsp + 2 * nb + 1) * 8;
        const double* gavg = nullptr;
        if (mode == 2) {
            d_sum_.reserve(ng * 8); d_avg_.reserve(ng * 8); d_avgdec_.reserve(ng * 8);
            HPM_DEV_CHECK(hipMemset(d_sum_.p, 0, ng * 8));
            hipLaunchKernelGGL(FusedCellKernel, dim3(grid), dim3(threads), shm, 0, (const double*)d_u_.p, ncells, nsp, nb, cpb, (const double*)d_R_.p, (const double*)d_P_.p, 3,
                               (const double*)nullptr, (const uint32_t*)d_vid_.p, (const uint32_t*)d_inv_.p, (double*)d_sum_.p, qb, 1.0 / qb, qres, 1.0 / qres,
                               (int32_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, (int*)d_flag_.p);
            hipLaunchKernelGGL(GroupMeanKernel, dim3(GridFor(ng)), dim3(kThreads), 0, 0, (const double*)d_sum_.p, (const uint32_t*)d_cnt_.p, ng, (double*)d_avg_.p);
            HPM_DEV_CHECK(hipGetLastError());
            hipLaunchKernelGGL(QuantizeKernel, dim3(GridFor(ng)), dim3(kThreads), 0, 0, (const double*)d_avg_.p, (const double*)nullptr, ng, 1.0 / qb, (int32_t*)d_kavg_.p, (int*)d_flag_.p);
            hipLaunchKernelGGL(DequantKernel, dim3(GridFor(ng)), dim3(kThreads), 0, 0, (const int32_t*)d_kavg_.p, ng, qb, (double*)d_avgdec_.p);
            HPM_DEV_CHECK(hipGetLastError());
            gavg = (const double*)d_avgdec_.p;
        }
        hipLaunchKernelGGL(FusedCellKernel, dim3(grid), dim3(threads), shm, 0, (const double*)d_u_.p, ncells, nsp, nb, cpb, (const double*)d_R_.p, (const double*)d_P_.p, mode,
                           gavg, (const uint32_t*)d_vid_.p, (const uint32_t*)d_inv_.p, (double*)nullptr, qb, 1.0 / qb, qres, 1.0 / qres,
                           (int32_t*)d_kavg_.p, (int32_t*)d_krem_.p, (int32_t*)d_k_.p, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        int overflow = 0; HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));   // the one synchronous read of the component
        if (overflow) return false;
        out.d_res = (const int32_t*)d_k_.p; out.n_res = npts; d_k_n_ = npts;
        if (mode == 0) { out.d_avg = (const int32_t*)d_krem_.p; out.n_avg = nbase; out.d_rem = nullptr; out.n_rem = 0; }
        else { out.d_avg = (const int32_t*)d_kavg_.p; out.n_avg = mode == 2 ? ng : ncells; out.d_rem = (const int32_t*)d_krem_.p; out.n_rem = (mode == 1 && nb == 1) ? 0 : nbase; }
        return true;
    }
    bool ReconstructFused(const OperatorSet& ops, size_t ncells, BaseMethod method, const DedupGeometry* geom, double qb, double qres, const int32_t* d_avg, const int32_t* d_rem, const int32_t* d_res, double* out) override {
        const int nsp = ops.nsp_data, nb = ops.nsp_base; const size_t npts = ncells * (size_t)nsp, nbase = ncells * (size_t)nb;
        const int cpb = CellsPerBlock(nsp, nb); if (cpb <= 0 || !d_res) return false;
        UploadDense(ops);
        const int mode = method == BaseMethod::Plain ? 0 : method == BaseMethod::CellAvg ? 1 : 2;
        const double* gavg = nullptr;
        if (mode == 2) {
            if (!geom || geom->vid.size() != nbase || !d_avg || !d_rem) return false;
            UploadGeometry(*geom, nbase);
            const size_t ng = geom->ngroups; d_avgdec_.reserve(ng * 8);
            hipLaunchKernelGGL(DequantKernel, dim3(GridFor(ng)), dim3(kThreads), 0, 0, d_avg, ng, qb, (double*)d_avgdec_.p);
            gavg = (const double*)d_avgdec_.p;
        } else if (mode == 1) { if (!d_avg) return false; if (nb > 1 && !d_rem) return false; if (nb == 1) d_rem = nullptr; }
        else { if (!d_avg) return false; d_rem = d_avg; }   // plain: the base symbols are the "avg" stream
        d_out_.reserve(npts * 8);
        const unsigned grid = static_cast<unsigned>((ncells + cpb - 1) / cpb); const unsigned threads = static_cast<unsigned>(cpb * nsp);
        hipLaunchKernelGGL(ReconstructFusedKernel, dim3(grid), dim3(threads), static_cast<size_t>(cpb) * nb * 8, 0, ncells, nsp, nb, cpb, (const double*)d_P_.p, mode,
                           d_avg, d_rem, d_res, gavg, (const uint32_t*)d_vid_.p, (const uint32_t*)d_inv_.p, qb, qres, (double*)d_out_.p);
        HPM_DEV_CHECK(hipGetLastError());
        HPM_DEV_CHECK(hipMemcpy(out, d_out_.p, npts * 8, hipMemcpyDeviceToHost));
        return true;
    }
    const int32_t* UploadSymbols(int slot, const int32_t* host, size_t n) override {
        if (slot < 0 || slot > 2) return nullptr;
        d_up_[slot].reserve(std::max<size_t>(n, 1) * 4);
        HPM_DEV_CHECK(hipMemcpy(d_up_[slot].p, host, n * 4, hipMemcpyHostToDevice));
        return static_cast<const int32_t*>(d_up_[slot].p);
    }
    const int32_t* KeepDeviceSymbols(int slot, const int32_t* d, size_t n) override {
        if (slot < 0 || slot > 2 || !d) return nullptr;
        d_up_[slot].reserve(std::max<size_t>(n, 1) * 4);
        HPM_DEV_CHECK(hipMemcpy(d_up_[slot].p, d, n * 4, hipMemcpyDeviceToDevice));
        return static_cast<const int32_t*>(d_up_[slot].p);
    }

    // ---- device-resident base path -----------------------------------------------------------------
    bool SupportsDeviceBase() const override { return true; }
    bool RestrictDevice(const OperatorSet& ops, size_t ncells, const double* u) override {
        const size_t npts = ncells * (size_t)ops.nsp_data, nbase = ncells * (size_t)ops.nsp_base;
        if (!(u == resident_ && resident_n_ == npts)) BeginComponent(u, npts);
        d_basev_.reserve(nbase * 8);
        Apply(ops, true, d_u_.p, ncells, d_basev_.p);
        nbase_ = nbase; d_basedec_valid_ = false;
        return true;
    }
    bool QuantizeToDevice(const double* d_x, size_t n, double q, int32_t* d_k, double* d_dec) {
        d_flag_.reserve(4); HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        hipLaunchKernelGGL(QuantizeKernel, dim3(GridFor(n)), dim3(kThreads), 0, 0, d_x, (const double*)nullptr, n, 1.0 / q, d_k, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        if (d_dec) { hipLaunchKernelGGL(DequantKernel, dim3(GridFor(n)), dim3(kThreads), 0, 0, (const int32_t*)d_k, n, q, d_dec); HPM_DEV_CHECK(hipGetLastError()); }
        int overflow = 0; HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        return overflow == 0;
    }
    bool SplitPlainDevice(size_t ncells, int nb, double qb, DeviceStreams& out) override {
        const size_t nbase = ncells * (size_t)nb;
        d_kavg_.reserve(nbase * 4); d_basedec_.reserve(nbase * 8);
        if (!QuantizeToDevice((const double*)d_basev_.p, nbase, qb, (int32_t*)d_kavg_.p, (double*)d_basedec_.p)) return false;
        out.d_avg = (const int32_t*)d_kavg_.p; out.n_avg = nbase; out.d_rem = nullptr; out.n_rem = 0;
        d_basedec_valid_ = true; return true;
    }
    bool SplitCellAvgDevice(size_t ncells, int nb, double qb, DeviceStreams& out) override {
        const size_t nbase = ncells * (size_t)nb;
        d_avg_.reserve(ncells * 8); d_avgdec_.reserve(ncells * 8); d_kavg_.reserve(ncells * 4); d_krem_.reserve(nbase * 4); d_basedec_.reserve(nbase * 8);
        hipLaunchKernelGGL(CellMeansKernel, dim3(GridFor(ncells)), dim3(kThreads), 0, 0, (const double*)d_basev_.p, ncells, nb, (double*)d_avg_.p);
        HPM_DEV_CHECK(hipGetLastError());
        if (!QuantizeToDevice((const double*)d_avg_.p, ncells, qb, (int32_t*)d_kavg_.p, (double*)d_avgdec_.p)) return false;
        d_flag_.reserve(4); HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        hipLaunchKernelGGL(CellAvgRemainderKernel, dim3(GridFor(nbase)), dim3(kThreads), 0, 0, (const double*)d_basev_.p, (const double*)d_avgdec_.p, nbase, nb, qb, 1.0 / qb, (int32_t*)d_krem_.p, (double*)d_basedec_.p, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        int overflow = 0; HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        if (overflow) return false;
        out.d_avg = (const int32_t*)d_kavg_.p; out.n_avg = ncells; out.d_rem = (const int32_t*)d_krem_.p; out.n_rem = nb > 1 ? nbase : 0;
        d_basedec_valid_ = true; return true;
    }
    bool SplitDedupDevice(size_t ncells, int nb, double qb, const DedupGeometry& geom, DeviceStreams& out) override {
        const size_t nbase = ncells * (size_t)nb; const size_t ng = geom.ngroups;
        if (geom.vid.size() != nbase || geom.perm.size() != nbase) return false;
        if (geom_key_ != &geom || geom_n_ != nbase) {   // upload the geometry once per cached geometry object
            std::vector<uint32_t> cnt(ng, 0); for (uint32_t v : geom.vid) ++cnt[v];
            d_vid_.reserve(nbase * 4); d_perm_.reserve(nbase * 4); d_cnt_.reserve(ng * 4);
            HPM_DEV_CHECK(hipMemcpy(d_vid_.p, geom.vid.data(), nbase * 4, hipMemcpyHostToDevice));
            HPM_DEV_CHECK(hipMemcpy(d_perm_.p, geom.perm.data(), nbase * 4, hipMemcpyHostToDevice));
            HPM_DEV_CHECK(hipMemcpy(d_cnt_.p, cnt.data(), ng * 4, hipMemcpyHostToDevice));
            geom_key_ = &geom; geom_n_ = nbase;
        }
        d_avg_.reserve(ng * 8); d_avgdec_.reserve(ng * 8); d_kavg_.reserve(ng * 4); d_krem_.reserve(nbase * 4); d_basedec_.reserve(nbase * 8); d_sum_.reserve(ng * 8);
        HPM_DEV_CHECK(hipMemset(d_sum_.p, 0, ng * 8));
        hipLaunchKernelGGL(GroupSumKernel, dim3(GridFor(nbase)), dim3(kThreads), 0, 0, (const double*)d_basev_.p, (const uint32_t*)d_vid_.p, nbase, (double*)d_sum_.p);
        hipLaunchKernelGGL(GroupMeanKernel, dim3(GridFor(ng)), dim3(kThreads), 0, 0, (const double*)d_sum_.p, (const uint32_t*)d_cnt_.p, ng, (double*)d_avg_.p);
        HPM_DEV_CHECK(hipGetLastError());
        if (!QuantizeToDevice((const double*)d_avg_.p, ng, qb, (int32_t*)d_kavg_.p, (double*)d_avgdec_.p)) return false;
        d_flag_.reserve(4); HPM_DEV_CHECK(hipMemset(d_flag_.p, 0, 4));
        hipLaunchKernelGGL(DedupRemainderKernel, dim3(GridFor(nbase)), dim3(kThreads), 0, 0, (const double*)d_basev_.p, (const double*)d_avgdec_.p, (const uint32_t*)d_vid_.p, (const uint32_t*)d_perm_.p, nbase, qb, 1.0 / qb, (int32_t*)d_krem_.p, (double*)d_basedec_.p, (int*)d_flag_.p);
        HPM_DEV_CHECK(hipGetLastError());
        int overflow = 0; HPM_DEV_CHECK(hipMemcpy(&overflow, d_flag_.p, 4, hipMemcpyDeviceToHost));
        if (overflow) return false;
        out.d_avg = (const int32_t*)d_kavg_.p; out.n_avg = ng; out.d_rem = (const int32_t*)d_krem_.p; out.n_rem = nbase;
        d_basedec_valid_ = true; return true;
    }
    void CopyDeviceToHost(const int32_t* d, size_t n, int32_t* host) const override { HPM_DEV_CHECK(hipMemcpy(host, d, n * 4, hipMemcpyDeviceToHost)); }

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
        const void* d_bd = nullptr;
        if (base_dec) { d_base_.reserve(nbase * 8); HPM_DEV_CHECK(hipMemcpy(d_base_.p, base_dec, nbase * 8, hipMemcpyHostToDevice)); d_bd = d_base_.p; }
        else if (d_basedec_valid_ && nbase_ == nbase) d_bd = d_basedec_.p;   // resident decoded base from Split*Device
        else throw std::runtime_error("ResidualFromBase: no decoded base available");
        d_out_.reserve(npts * 8);
        Apply(ops, false, d_bd, ncells, d_out_.p);                       // d_out_ = P base_dec (stays on the device)
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
    DeviceBuffer d_basev_, d_basedec_, d_avg_, d_avgdec_, d_sum_, d_kavg_, d_krem_, d_vid_, d_perm_, d_cnt_;   // device base path
    DeviceBuffer d_inv_, d_R_, d_P_, d_partial_, d_up_[3]; std::string dense_key_;                           // fused path
    const DedupGeometry* geom_key_ = nullptr; size_t geom_n_ = 0;
    size_t nbase_ = 0; bool d_basedec_valid_ = false;
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
    void ComponentStats(const double* u, size_t n, double& lo, double& hi, double& ss) override { b_.ComponentStats(u, n, lo, hi, ss); }
    bool SupportsFused(const OperatorSet& o) const override { return b_.SupportsFused(o); }
    bool FusedCompress(const OperatorSet& o, size_t n, BaseMethod m, const DedupGeometry* g, double qb, double qr, const double* u, FusedStreams& out) override { return b_.FusedCompress(o, n, m, g, qb, qr, u, out); }
    bool ReconstructFused(const OperatorSet& o, size_t n, BaseMethod m, const DedupGeometry* g, double qb, double qr, const int32_t* a, const int32_t* r, const int32_t* s, double* out) override { return b_.ReconstructFused(o, n, m, g, qb, qr, a, r, s, out); }
    const int32_t* UploadSymbols(int slot, const int32_t* host, size_t n) override { return b_.UploadSymbols(slot, host, n); }
    const int32_t* KeepDeviceSymbols(int slot, const int32_t* d, size_t n) override { return b_.KeepDeviceSymbols(slot, d, n); }
    bool SupportsDeviceBase() const override { return true; }
    bool RestrictDevice(const OperatorSet& o, size_t n, const double* u) override { return b_.RestrictDevice(o, n, u); }
    bool SplitCellAvgDevice(size_t n, int nb, double qb, DeviceStreams& out) override { return b_.SplitCellAvgDevice(n, nb, qb, out); }
    bool SplitDedupDevice(size_t n, int nb, double qb, const DedupGeometry& g, DeviceStreams& out) override { return b_.SplitDedupDevice(n, nb, qb, g, out); }
    bool SplitPlainDevice(size_t n, int nb, double qb, DeviceStreams& out) override { return b_.SplitPlainDevice(n, nb, qb, out); }
    void CopyDeviceToHost(const int32_t* d, size_t n, int32_t* host) const override { b_.CopyDeviceToHost(d, n, host); }
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
