// GPU translation unit of the MGARD-X lossless stage, shared by HIP and CUDA. Included from
// MgardXHuff_HIP.hip / MgardXHuff_CUDA.cu after DeviceShim.h with HPM_XDEVICE (mgard_x::HIP / CUDA) and
// HPM_XNS (namespace hip / cuda) defined. Device-resident encode/decode: zigzag, range check and the
// inverse mapping run on the device; only compressed bytes cross the link.
#include "hpmgard/codec/MgardXHuffImpl.hpp"

namespace hpmgard {
namespace codec {
namespace xhuff {

template <> bool DeviceErrorCheck<HPM_XDEVICE>() {
    const hipError_t sync = hipDeviceSynchronize();
    const hipError_t last = hipGetLastError();
    if (sync == hipSuccess && last == hipSuccess) return true;
    std::cerr << "[hpmgard xhuff] device error: " << hipGetErrorString(sync != hipSuccess ? sync : last) << "\n";
    return false;
}

namespace HPM_XNS {

namespace {
using I = Impl<HPM_XDEVICE>;

// zigzag into `out`; minmax[0] = max code (init 0), minmax[1] = min code (init 0xFFFFFFFF)
__global__ void ZigzagMinMaxKernel(const int32_t* __restrict__ in, int32_t* __restrict__ out, size_t n, unsigned* minmax) {
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
__global__ void UnzigzagKernel(const int32_t* __restrict__ in, int32_t* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t z = static_cast<uint32_t>(in[i]);
        out[i] = static_cast<int32_t>((z >> 1) ^ (0u - (z & 1u)));
    }
}
__global__ void CountMismatchKernel(const int32_t* __restrict__ a, const int32_t* __restrict__ b, size_t n, unsigned* count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) atomicAdd(count, 1u);
}
// Process-lifetime scratch (never freed: freeing in a static destructor after runtime teardown is unsafe).
struct Scratch { int32_t* codes = nullptr; int32_t* dec = nullptr; size_t cap = 0; unsigned* scalar = nullptr; };   // scalar: [0] max, [1] min, [2] mismatches
Scratch& GetScratch(size_t n) {
    static Scratch s;
    if (!s.scalar) { if (hipMalloc((void**)&s.scalar, 16) != hipSuccess) throw std::runtime_error("device scratch allocation failed"); }
    if (n > s.cap) {
        if (s.codes) (void)hipFree(s.codes); if (s.dec) (void)hipFree(s.dec); s.codes = s.dec = nullptr; s.cap = 0;
        if (hipMalloc((void**)&s.codes, n * sizeof(int32_t)) != hipSuccess || hipMalloc((void**)&s.dec, n * sizeof(int32_t)) != hipSuccess) throw std::runtime_error("device code buffer allocation failed");
        s.cap = n;
    }
    return s;
}
inline unsigned Blocks(size_t n) { const size_t b = (n + 255) / 256; return static_cast<unsigned>(std::min<size_t>(b, 4096)); }
} // namespace

bool DevicePresent() { int count = 0; return hipGetDeviceCount(&count) == hipSuccess && count > 0; }

bool Encode(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) { return I::Encode(codes, t, level, dict, block, out); }
bool Decode(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block) { return I::Decode(b, nb, codes, n, t, level, dict, block); }

// Symbols already on the device: zigzag + range check on the device, wrap without copying, compress with
// the persistent object, optional device compare.
bool EncodeDevice(const int32_t* d_sym, size_t n, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) {
    I::InitOnce();
    std::lock_guard<std::mutex> lk(I::Mutex());
    try {
        Scratch& s = GetScratch(n);
        const unsigned init[3] = {0u, 0xFFFFFFFFu, 0u};
        if (hipMemcpy(s.scalar, init, 12, hipMemcpyHostToDevice) != hipSuccess) return false;
        hipLaunchKernelGGL(ZigzagMinMaxKernel, dim3(Blocks(n)), dim3(256), 0, 0, d_sym, s.codes, n, s.scalar);
        unsigned mm[2] = {0, 0};
        if (hipMemcpy(mm, s.scalar, 8, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (!DeviceErrorCheck<HPM_XDEVICE>()) return false;
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
            if (!DeviceErrorCheck<HPM_XDEVICE>()) return false;
            if (mism != 0) { std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " device stream failed the decode check (" << mism << " symbols); using the fallback coder\n"; return false; }
        }
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " device compress failed: " << ex.what() << "\n";
        return false;
    }
}

// Decode on the device and leave the (un-zigzagged) symbols there; the pointer is valid until the next
// call of this stage. nullptr on failure.
const int32_t* DecodeDevice(const uint8_t* b, size_t nb, size_t n, XType t, int level, uint32_t dict, uint32_t block) {
    I::InitOnce();
    std::lock_guard<std::mutex> lk(I::Mutex());
    try {
        Scratch& s = GetScratch(n);
        I::Entry& e = I::Get(t, level, dict, block, static_cast<mgard_x::SIZE>(n));
        I::ByteArray comp({static_cast<mgard_x::SIZE>(nb)}); comp.load(reinterpret_cast<const mgard_x::Byte*>(b));
        I::SymArray dec({static_cast<mgard_x::SIZE>(n)}, s.dec);         // external; MGARD-X may still reallocate, so read dec.data() below
        if (!I::DecompressArray(e, comp, dec)) return nullptr;
        hipLaunchKernelGGL(UnzigzagKernel, dim3(Blocks(n)), dim3(256), 0, 0, dec.data(), s.codes, n);
        if (!DeviceErrorCheck<HPM_XDEVICE>()) return nullptr;
        return s.codes;
    } catch (const std::exception& ex) {
        std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " device decompress failed: " << ex.what() << "\n";
        return nullptr;
    }
}

void Warmup(XType t, int level) {
    const size_t n = size_t(1) << 17;
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (i % 11 == 0) ? static_cast<int32_t>((i / 11) % 61) - 30 : 0;
    int32_t* d = nullptr;
    if (hipMalloc((void**)&d, n * sizeof(int32_t)) != hipSuccess) return;
    if (hipMemcpy(d, v.data(), n * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess) {
        std::vector<uint8_t> out;
        if (EncodeDevice(d, n, t, level, DictSize(), BlockSize(), out) && out.size() > 8) (void)DecodeDevice(out.data(), out.size(), n, t, level, DictSize(), BlockSize());
    }
    (void)hipFree(d);
}

} // namespace HPM_XNS
} // namespace xhuff
} // namespace codec
} // namespace hpmgard
