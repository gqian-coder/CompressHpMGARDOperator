// GPU FRBP stage shared by HIP and CUDA (included from Frbp_HIP.hip / Frbp_CUDA.cu after DeviceShim.h with
// HPM_FNS = hip | cuda). Byte-identical to the CPU reference in RawFrbpZstd.cpp.
#include "hpmgard/codec/Frbp.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpmgard {
namespace codec {
namespace frbp {
namespace HPM_FNS {

namespace {
constexpr uint32_t kMagic = 0x46524250u, kVersion = 1;
constexpr int kScan = 1024;

#define HPM_FRBP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) throw std::runtime_error(std::string("frbp device error: ") + hipGetErrorString(e_)); } while (0)

struct Scratch {   // process lifetime (never freed: unsafe after runtime teardown)
    uint8_t* rate = nullptr; uint32_t* words = nullptr; uint32_t* off = nullptr; uint32_t* chunk = nullptr; size_t nblocks_cap = 0, nchunk_cap = 0;
    uint8_t* buf = nullptr; size_t buf_cap = 0;          // [rates padded][body] for encode, input stream for decode
    int32_t* dec = nullptr; size_t dec_cap = 0;
};
Scratch& GetScratch(size_t nblocks, size_t buf_bytes, size_t ndec) {
    static Scratch s;
    const size_t nchunk = (nblocks + kScan - 1) / kScan;
    if (nblocks > s.nblocks_cap) {
        if (s.rate) { (void)hipFree(s.rate); (void)hipFree(s.words); (void)hipFree(s.off); }
        HPM_FRBP_CHECK(hipMalloc((void**)&s.rate, nblocks + 4)); HPM_FRBP_CHECK(hipMalloc((void**)&s.words, nblocks * 4)); HPM_FRBP_CHECK(hipMalloc((void**)&s.off, nblocks * 4));
        s.nblocks_cap = nblocks;
    }
    if (nchunk > s.nchunk_cap) { if (s.chunk) (void)hipFree(s.chunk); HPM_FRBP_CHECK(hipMalloc((void**)&s.chunk, (nchunk + 1) * 4)); s.nchunk_cap = nchunk; }
    if (buf_bytes > s.buf_cap) { if (s.buf) (void)hipFree(s.buf); HPM_FRBP_CHECK(hipMalloc((void**)&s.buf, buf_bytes)); s.buf_cap = buf_bytes; }
    if (ndec > s.dec_cap) { if (s.dec) (void)hipFree(s.dec); HPM_FRBP_CHECK(hipMalloc((void**)&s.dec, ndec * 4)); s.dec_cap = ndec; }
    return s;
}
inline unsigned Blocks(size_t n, int t) { return static_cast<unsigned>((n + t - 1) / t); }

__device__ __forceinline__ int BitsNeeded(uint32_t v) { return v ? 32 - __clz(v) : 0; }

// one thread per 32-symbol block: rate byte and word count (0 or 1 + rate)
__global__ void RateKernel(const int32_t* __restrict__ sym, size_t n, size_t nblocks, uint8_t* __restrict__ rate, uint32_t* __restrict__ words) {
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    uint32_t mx = 0;
    for (int i = 0; i < 32; ++i) { const size_t idx = b * 32 + i; const int32_t v = idx < n ? sym[idx] : 0; const uint32_t a = v < 0 ? (0u - static_cast<uint32_t>(v)) : static_cast<uint32_t>(v); mx = max(mx, a); }
    const int r = BitsNeeded(mx);
    rate[b] = static_cast<uint8_t>(r); words[b] = r ? 1u + r : 0u;
}
__global__ void WordsFromRateKernel(const uint8_t* __restrict__ rate, size_t nblocks, uint32_t* __restrict__ words) {
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b < nblocks) { const uint32_t r = rate[b]; words[b] = r ? 1u + r : 0u; }
}
// exclusive scan: per-chunk scan + chunk totals, totals scan (single block), add back
__global__ void ChunkScanKernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint32_t* __restrict__ sums, size_t n) {
    __shared__ uint32_t s[kScan];
    const size_t i = blockIdx.x * (size_t)kScan + threadIdx.x;
    const uint32_t v = i < n ? in[i] : 0u;
    s[threadIdx.x] = v; __syncthreads();
    for (int d = 1; d < kScan; d <<= 1) { const uint32_t t = threadIdx.x >= (unsigned)d ? s[threadIdx.x - d] : 0u; __syncthreads(); s[threadIdx.x] += t; __syncthreads(); }
    if (i < n) out[i] = s[threadIdx.x] - v;
    if (threadIdx.x == kScan - 1) sums[blockIdx.x] = s[kScan - 1];
}
__global__ void TotalsScanKernel(uint32_t* __restrict__ sums, size_t nchunk) {   // in place -> exclusive prefix, sums[nchunk] = total
    __shared__ uint32_t s[kScan]; __shared__ uint32_t carry;
    if (threadIdx.x == 0) carry = 0; __syncthreads();
    for (size_t base = 0; base < nchunk; base += kScan) {
        const size_t i = base + threadIdx.x; const uint32_t v = i < nchunk ? sums[i] : 0u;
        s[threadIdx.x] = v; __syncthreads();
        for (int d = 1; d < kScan; d <<= 1) { const uint32_t t = threadIdx.x >= (unsigned)d ? s[threadIdx.x - d] : 0u; __syncthreads(); s[threadIdx.x] += t; __syncthreads(); }
        const uint32_t c = carry; __syncthreads();
        if (i < nchunk) sums[i] = c + s[threadIdx.x] - v;
        if (threadIdx.x == kScan - 1) carry = c + s[kScan - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[nchunk] = carry;
}
__global__ void AddChunkKernel(uint32_t* __restrict__ out, const uint32_t* __restrict__ prefix, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] += prefix[i / kScan];
}
// one thread per block: sign word + planes at word offset off[b]
__global__ void PackKernel(const int32_t* __restrict__ sym, size_t n, size_t nblocks, const uint8_t* __restrict__ rate, const uint32_t* __restrict__ off, uint32_t* __restrict__ body) {
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const int r = rate[b]; if (!r) return;
    uint32_t absv[32]; uint32_t sign = 0;
    for (int i = 0; i < 32; ++i) { const size_t idx = b * 32 + i; const int32_t v = idx < n ? sym[idx] : 0; if (v < 0) { sign |= 1u << i; absv[i] = 0u - static_cast<uint32_t>(v); } else absv[i] = static_cast<uint32_t>(v); }
    uint32_t* w = body + off[b];
    w[0] = sign;
    for (int bit = 0; bit < r; ++bit) { uint32_t plane = 0; for (int i = 0; i < 32; ++i) plane |= ((absv[i] >> bit) & 1u) << i; w[1 + bit] = plane; }
}
// one thread per symbol
__global__ void UnpackKernel(const uint8_t* __restrict__ rate, const uint32_t* __restrict__ off, const uint32_t* __restrict__ body, size_t n, int32_t* __restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t b = i >> 5; const int lane = static_cast<int>(i & 31);
    const int r = rate[b];
    if (!r) { out[i] = 0; return; }
    const uint32_t* w = body + off[b];
    uint32_t val = 0; for (int bit = 0; bit < r; ++bit) val |= ((w[1 + bit] >> lane) & 1u) << bit;
    out[i] = ((w[0] >> lane) & 1u) ? -static_cast<int32_t>(val) : static_cast<int32_t>(val);
}
void Scan(Scratch& s, size_t nblocks) {
    const size_t nchunk = (nblocks + kScan - 1) / kScan;
    hipLaunchKernelGGL(ChunkScanKernel, dim3((unsigned)nchunk), dim3(kScan), 0, 0, s.words, s.off, s.chunk, nblocks);
    hipLaunchKernelGGL(TotalsScanKernel, dim3(1), dim3(kScan), 0, 0, s.chunk, nchunk);
    hipLaunchKernelGGL(AddChunkKernel, dim3(Blocks(nblocks, 256)), dim3(256), 0, 0, s.off, s.chunk, nblocks);
    HPM_FRBP_CHECK(hipGetLastError());
}
inline void PutU32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i)); }
inline void PutU64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i)); }
inline uint32_t GetU32(const uint8_t* p) { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i); return v; }
inline uint64_t GetU64(const uint8_t* p) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i); return v; }
} // namespace

bool DevicePresent() { int count = 0; return hipGetDeviceCount(&count) == hipSuccess && count > 0; }

bool EncodeDevice(const int32_t* d_sym, size_t n, double eb_info, std::vector<uint8_t>& out) {
    try {
        const size_t nblocks = (n + 31) / 32, rate_padded = (nblocks + 3) / 4 * 4;
        Scratch& s = GetScratch(nblocks, 0, 0);
        hipLaunchKernelGGL(RateKernel, dim3(Blocks(nblocks, 256)), dim3(256), 0, 0, d_sym, n, nblocks, s.rate, s.words);
        HPM_FRBP_CHECK(hipGetLastError());
        Scan(s, nblocks);
        const size_t nchunk = (nblocks + kScan - 1) / kScan;
        uint32_t total_words = 0;
        HPM_FRBP_CHECK(hipMemcpy(&total_words, s.chunk + nchunk, 4, hipMemcpyDeviceToHost));   // single sync: body size
        const size_t body_bytes = static_cast<size_t>(total_words) * 4;
        Scratch& s2 = GetScratch(nblocks, rate_padded + body_bytes, 0);
        HPM_FRBP_CHECK(hipMemset(s2.buf + nblocks, 0, rate_padded - nblocks));
        HPM_FRBP_CHECK(hipMemcpy(s2.buf, s2.rate, nblocks, hipMemcpyDeviceToDevice));
        if (total_words) {
            hipLaunchKernelGGL(PackKernel, dim3(Blocks(nblocks, 256)), dim3(256), 0, 0, d_sym, n, nblocks, s2.rate, s2.off, reinterpret_cast<uint32_t*>(s2.buf + rate_padded));
            HPM_FRBP_CHECK(hipGetLastError());
        }
        out.resize(40 + rate_padded + body_bytes);
        uint8_t* h = out.data();
        PutU32(h, kMagic); PutU32(h + 4, kVersion); PutU64(h + 8, n); std::memcpy(h + 16, &eb_info, 8); PutU64(h + 24, nblocks); PutU64(h + 32, body_bytes);
        HPM_FRBP_CHECK(hipMemcpy(h + 40, s2.buf, rate_padded + body_bytes, hipMemcpyDeviceToHost));
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[hpmgard frbp] device encode failed: %s\n", e.what());
        return false;
    }
}

const int32_t* DecodeDevice(const uint8_t* b, size_t nb, size_t n) {
    try {
        if (nb < 40 || GetU32(b) != kMagic || GetU32(b + 4) != kVersion) return nullptr;
        const uint64_t nn = GetU64(b + 8), nblocks = GetU64(b + 24), body_bytes = GetU64(b + 32);
        if (nn != n || nblocks != (n + 31) / 32) return nullptr;
        const size_t rate_padded = (nblocks + 3) / 4 * 4;
        if (nb < 40 + rate_padded + body_bytes) return nullptr;
        Scratch& s = GetScratch(nblocks, rate_padded + body_bytes, n);
        HPM_FRBP_CHECK(hipMemcpy(s.buf, b + 40, rate_padded + body_bytes, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(WordsFromRateKernel, dim3(Blocks(nblocks, 256)), dim3(256), 0, 0, s.buf, nblocks, s.words);
        HPM_FRBP_CHECK(hipGetLastError());
        Scan(s, nblocks);
        hipLaunchKernelGGL(UnpackKernel, dim3(Blocks(n, 256)), dim3(256), 0, 0, s.buf, s.off, reinterpret_cast<const uint32_t*>(s.buf + rate_padded), n, s.dec);
        HPM_FRBP_CHECK(hipGetLastError());
        HPM_FRBP_CHECK(hipDeviceSynchronize());
        return s.dec;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[hpmgard frbp] device decode failed: %s\n", e.what());
        return nullptr;
    }
}

} // namespace HPM_FNS
} // namespace frbp
} // namespace codec
} // namespace hpmgard
