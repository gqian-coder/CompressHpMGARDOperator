// HIP instantiation of the MGARD-X lossless stage (compiled as HIP with MGARDX_COMPILE_HIP), plus the
// device-resident encode path: zigzag + dictionary check on the device, no host copy of the symbols.
#include "hpmgard/codec/MgardXHuffImpl.hpp"
#include <hip/hip_runtime.h>

namespace hpmgard {
namespace codec {
namespace xhuff {

template <> bool DeviceErrorCheck<mgard_x::HIP>() {
    const hipError_t sync = hipDeviceSynchronize();
    const hipError_t last = hipGetLastError();
    if (sync == hipSuccess && last == hipSuccess) return true;
    std::cerr << "[hpmgard xhuff] HIP error: " << hipGetErrorString(sync != hipSuccess ? sync : last) << "\n";
    return false;
}

namespace {
// zigzag in place into `out`; minmax[0] = max code (init 0), minmax[1] = min code (init 0xFFFFFFFF)
__global__ void ZigzagMinMaxKernel(const int32_t* __restrict in, int32_t* __restrict out, size_t n, unsigned* minmax) {
    __shared__ unsigned smx[256]; __shared__ unsigned smn[256];
    unsigned lmax = 0, lmin = 0xFFFFFFFFu;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t z = (static_cast<uint32_t>(in[i]) << 1) ^ static_cast<uint32_t>(in[i] >> 31);
        out[i] = static_cast<int32_t>(z);
        lmax = max(lmax, z); lmin = min(lmin, z);
    }
    smx[threadIdx.x] = lmax; smn[threadIdx.x] = lmin;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smx[threadIdx.x] = max(smx[threadIdx.x], smx[threadIdx.x + s]); smn[threadIdx.x] = min(smn[threadIdx.x], smn[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicMax(minmax, smx[0]); atomicMin(minmax + 1, smn[0]); }
}
__global__ void CountMismatchKernel(const int32_t* __restrict a, const int32_t* __restrict b, size_t n, unsigned* count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) atomicAdd(count, 1u);
}
// Process-lifetime scratch (never freed: freeing in a static destructor after HIP teardown is unsafe).
struct Scratch { int32_t* codes = nullptr; size_t cap = 0; unsigned* scalar = nullptr; };   // scalar: [0] max, [1] min, [2] mismatches
Scratch& GetScratch(size_t n) {
    static Scratch s;
    if (!s.scalar) { if (hipMalloc(&s.scalar, 16) != hipSuccess) throw std::runtime_error("hipMalloc scratch"); }
    if (n > s.cap) { if (s.codes) (void)hipFree(s.codes); s.codes = nullptr; s.cap = 0; if (hipMalloc(&s.codes, n * sizeof(int32_t)) != hipSuccess) throw std::runtime_error("hipMalloc codes"); s.cap = n; }
    return s;
}
inline unsigned Blocks(size_t n) { const size_t b = (n + 255) / 256; return static_cast<unsigned>(std::min<size_t>(b, 4096)); }
} // namespace

bool HipDevicePresent() { int count = 0; return hipGetDeviceCount(&count) == hipSuccess && count > 0; }

bool EncodeHip(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) { return Impl<mgard_x::HIP>::Encode(codes, t, level, dict, block, out); }
bool DecodeHip(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block) { return Impl<mgard_x::HIP>::Decode(b, nb, codes, n, t, level, dict, block); }

// Symbols already on the device: zigzag + max on the device, wrap the result without copying, compress
// with the persistent object, verify with a device compare.
bool EncodeHipDevice(const int32_t* d_sym, size_t n, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) {
    using I = Impl<mgard_x::HIP>;
    I::InitOnce();
    std::lock_guard<std::mutex> lk(I::Mutex());
    try {
        Scratch& s = GetScratch(n);
        const unsigned init[3] = {0u, 0xFFFFFFFFu, 0u};
        if (hipMemcpy(s.scalar, init, 12, hipMemcpyHostToDevice) != hipSuccess) return false;
        hipLaunchKernelGGL(ZigzagMinMaxKernel, dim3(Blocks(n)), dim3(256), 0, 0, d_sym, s.codes, n, s.scalar);
        unsigned mm[2] = {0, 0};
        if (hipMemcpy(mm, s.scalar, 8, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (!DeviceErrorCheck<mgard_x::HIP>()) return false;
        if (mm[0] >= dict) return false;            // outside the MGARD-X dictionary: let the caller fall back
        if (!AlphabetOk(t, mm[1], mm[0])) return false;
        I::Entry& e = I::Get(t, level, dict, block, static_cast<mgard_x::SIZE>(n));
        I::SymArray arr({static_cast<mgard_x::SIZE>(n)}, s.codes);   // external allocation: no copy, not freed
        if (!I::CompressArray(e, arr, out)) return false;
        if (VerifyEnabled()) {
            I::ByteArray comp({static_cast<mgard_x::SIZE>(out.size())}); comp.load(out.data());
            I::SymArray dec({static_cast<mgard_x::SIZE>(n)});
            if (!I::DecompressArray(e, comp, dec)) return false;
            hipLaunchKernelGGL(CountMismatchKernel, dim3(Blocks(n)), dim3(256), 0, 0, dec.data(), s.codes, n, s.scalar + 2);
            unsigned mism = 0;
            if (hipMemcpy(&mism, s.scalar + 2, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
            if (!DeviceErrorCheck<mgard_x::HIP>()) return false;
            if (mism != 0) { std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " device stream failed the decode check (" << mism << " symbols); using the fallback coder\n"; return false; }
        }
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " device compress failed: " << ex.what() << "\n";
        return false;
    }
}

void WarmupHip(XType t, int level) {
    const size_t n = size_t(1) << 17;
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (i % 11 == 0) ? static_cast<int32_t>((i / 11) % 61) - 30 : 0;
    int32_t* d = nullptr;
    if (hipMalloc(&d, n * sizeof(int32_t)) != hipSuccess) return;
    if (hipMemcpy(d, v.data(), n * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess) {
        std::vector<uint8_t> out;
        (void)EncodeHipDevice(d, n, t, level, DictSize(), BlockSize(), out);
    }
    (void)hipFree(d);
}

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
