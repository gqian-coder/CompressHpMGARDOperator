#include "hpmgard/codec/Codec.h"
#include "hpmgard/Bytes.h"
#include <zstd.h>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace hpmgard {
namespace codec {

// ---------------------------------------------------------------- raw
std::vector<uint8_t> RawEncode(const int32_t* sym, size_t n) {
    std::vector<uint8_t> out(n * 4);
    for (size_t i = 0; i < n; ++i) { const uint32_t v = static_cast<uint32_t>(sym[i]); for (int k = 0; k < 4; ++k) out[4 * i + k] = static_cast<uint8_t>(v >> (8 * k)); }
    return out;
}
void RawDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n) {
    if (nb < n * 4) throw std::runtime_error("RawDecode: truncated");
    for (size_t i = 0; i < n; ++i) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= static_cast<uint32_t>(b[4 * i + k]) << (8 * k); out[i] = static_cast<int32_t>(v); }
}

// ---------------------------------------------------------------- FRBP (mp-MGARD fixed-rate format)
namespace {
constexpr uint32_t kFrbpMagic = 0x46524250u;   // "FRBP"
constexpr uint32_t kFrbpVersion = 1;
inline int BitsNeeded(uint32_t v) { int r = 0; while (v) { ++r; v >>= 1; } return r; }
}

std::vector<uint8_t> FrbpEncode(const int32_t* sym, size_t n, double eb_info) {
    const size_t nblocks = (n + 31) / 32;
    std::vector<uint8_t> rate(nblocks);
    std::vector<uint32_t> absv(32);
    // pass 1: rates and body size
    uint64_t body = 0;
    for (size_t bI = 0; bI < nblocks; ++bI) {
        uint32_t mx = 0;
        for (int i = 0; i < 32; ++i) { const size_t idx = bI * 32 + i; const int32_t v = idx < n ? sym[idx] : 0; const uint32_t a = v < 0 ? (0u - static_cast<uint32_t>(v)) : static_cast<uint32_t>(v); if (a > mx) mx = a; }
        rate[bI] = static_cast<uint8_t>(BitsNeeded(mx));
        if (rate[bI]) body += 4 + 4ull * rate[bI];
    }
    const size_t rate_padded = (nblocks + 3) / 4 * 4;
    ByteWriter w;
    w.u32(kFrbpMagic); w.u32(kFrbpVersion); w.u64(n); w.f64(eb_info); w.u64(nblocks); w.u64(body);
    w.bytes(rate.data(), nblocks);
    for (size_t k = nblocks; k < rate_padded; ++k) w.u8(0);
    w.buf.reserve(w.buf.size() + body);
    for (size_t bI = 0; bI < nblocks; ++bI) {
        const int r = rate[bI];
        if (!r) continue;
        uint32_t sign = 0;
        for (int i = 0; i < 32; ++i) {
            const size_t idx = bI * 32 + i; const int32_t v = idx < n ? sym[idx] : 0;
            if (v < 0) { sign |= 1u << i; absv[i] = 0u - static_cast<uint32_t>(v); } else absv[i] = static_cast<uint32_t>(v);
        }
        w.u32(sign);
        for (int b = 0; b < r; ++b) { uint32_t plane = 0; for (int i = 0; i < 32; ++i) plane |= ((absv[i] >> b) & 1u) << i; w.u32(plane); }
    }
    return w.buf;
}

void FrbpDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n) {
    ByteReader r(b, nb);
    if (r.u32() != kFrbpMagic) throw std::runtime_error("FrbpDecode: bad magic");
    if (r.u32() != kFrbpVersion) throw std::runtime_error("FrbpDecode: unsupported version");
    const uint64_t nn = r.u64(); r.f64(); const uint64_t nblocks = r.u64(); r.u64();
    if (nn != n) throw std::runtime_error("FrbpDecode: count mismatch");
    const uint8_t* rate = r.bytes(nblocks);
    r.skip((nblocks + 3) / 4 * 4 - nblocks);
    for (size_t bI = 0; bI < nblocks; ++bI) {
        const int rt = rate[bI];
        if (!rt) { for (int i = 0; i < 32; ++i) { const size_t idx = bI * 32 + i; if (idx < n) out[idx] = 0; } continue; }
        const uint32_t sign = r.u32();
        uint32_t absv[32] = {0};
        for (int bit = 0; bit < rt; ++bit) { const uint32_t plane = r.u32(); for (int i = 0; i < 32; ++i) absv[i] |= ((plane >> i) & 1u) << bit; }
        for (int i = 0; i < 32; ++i) { const size_t idx = bI * 32 + i; if (idx < n) out[idx] = (sign >> i) & 1u ? -static_cast<int32_t>(absv[i]) : static_cast<int32_t>(absv[i]); }
    }
}

// ---------------------------------------------------------------- zstd
namespace {
// Multithreaded zstd for large inputs: HPMGARD_ZSTD_WORKERS (default min(OpenMP threads, 8); 0 or 1 =
// single-threaded) and HPMGARD_ZSTD_JOB_MB (default 1, the independent job size; smaller = more parallel
// jobs, slightly less context for the matcher). Needs libzstd built with ZSTD_MULTITHREAD_SUPPORT; when
// the worker parameter is rejected the single-threaded path is used.
// No OpenMP call here: this runs on std::async worker threads too, and the Cray OpenMP runtime aborts on
// omp_get_max_threads() from a non-OpenMP thread. OMP_NUM_THREADS is honoured when set.
int ZstdWorkers() {
    static const int w = [] {
        const char* e = std::getenv("HPMGARD_ZSTD_WORKERS");
        if (e) return std::max(0, std::atoi(e));
        const char* o = std::getenv("OMP_NUM_THREADS");
        const int n = o ? std::atoi(o) : static_cast<int>(std::thread::hardware_concurrency());
        return std::max(1, std::min(n > 0 ? n : 8, 8));
    }();
    return w;
}
// Below this input size the single-threaded path is used: independent jobs cost about 1.4% in size and the
// stage takes only a few ms there (an 8 MB Huffman stream compresses in ~10 ms); the gain shows on the
// 100+ MB raw symbol streams (0.10 s -> 0.034 s on 8 workers). HPMGARD_ZSTD_MT_MIN_MB, default 16.
size_t ZstdMtMinBytes() {
    static const size_t m = [] { const char* e = std::getenv("HPMGARD_ZSTD_MT_MIN_MB"); const long mb = e ? std::atol(e) : 16; return static_cast<size_t>(std::max(0l, mb)) << 20; }();
    return m;
}
size_t ZstdJobBytes() {
    static const size_t j = [] { const char* e = std::getenv("HPMGARD_ZSTD_JOB_MB"); const long mb = e ? std::atol(e) : 1; return static_cast<size_t>(std::max(1l, mb)) << 20; }();
    return j;
}
}

std::vector<uint8_t> ZstdEncode(const uint8_t* b, size_t n, int level) {
    std::vector<uint8_t> out(ZSTD_compressBound(n));
    size_t c = 0; bool done = false;
    const int workers = ZstdWorkers();
    if (workers > 1 && n >= ZstdMtMinBytes()) {
        static thread_local ZSTD_CCtx* cctx = nullptr;   // reused so the worker pool is created once per thread
        if (!cctx) cctx = ZSTD_createCCtx();
        if (cctx) {
            ZSTD_CCtx_reset(cctx, ZSTD_reset_session_and_parameters);
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
            if (!ZSTD_isError(ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers))) {
                ZSTD_CCtx_setParameter(cctx, ZSTD_c_jobSize, static_cast<int>(std::min<size_t>(ZstdJobBytes(), size_t(1) << 30)));
                c = ZSTD_compress2(cctx, out.data(), out.size(), b, n);
                done = !ZSTD_isError(c);
            }
        }
    }
    if (!done) c = ZSTD_compress(out.data(), out.size(), b, n, level);
    if (ZSTD_isError(c)) throw std::runtime_error(std::string("ZstdEncode: ") + ZSTD_getErrorName(c));
    out.resize(c);
    return out;
}
void ZstdDecode(const uint8_t* b, size_t nb, uint8_t* out, size_t n) {
    const size_t d = ZSTD_decompress(out, n, b, nb);
    if (ZSTD_isError(d)) throw std::runtime_error(std::string("ZstdDecode: ") + ZSTD_getErrorName(d));
    if (d != n) throw std::runtime_error("ZstdDecode: size mismatch");
}

} // namespace codec
} // namespace hpmgard
