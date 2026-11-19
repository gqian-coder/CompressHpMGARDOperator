// Template shared by the SERIAL and HIP translation units of the MGARD-X lossless stage.
// Included only from those two files (each compiled for its own MGARD-X backend).
#pragma once
#include "hpmgard/codec/MgardXHuff.h"
#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

// Backend error check after the MGARD-X calls (HIP: hipDeviceSynchronize/hipGetLastError; SERIAL: true).
// Specialised in each translation unit before Impl<Device> is instantiated.
template <typename Device> bool DeviceErrorCheck();

// HPMGARD_XHUFF_VERIFY=1 enables a decode-and-compare check after every encode (MGARD-X's decoder is not
// robust against a corrupt stream). Default off: the MGARD-X stages only run on a device that passed
// mgardx::DeviceWorks(), which already catches the known broken-encoder case, and the check costs about
// as much as the in-house coder (0.11 s for 33M symbols on an MI250X).
inline bool VerifyEnabled() {
    static const bool on = [] { const char* e = std::getenv("HPMGARD_XHUFF_VERIFY"); return e && e[0] == '1'; }();
    return on;
}

inline mgard_x::lossless_type ToLossless(XType t) {
    switch (t) {
    case XType::Huffman: return mgard_x::lossless_type::Huffman;
    case XType::HuffmanZstd: return mgard_x::lossless_type::Huffman_Zstd;
    case XType::ZeroRleRans: return mgard_x::lossless_type::ZeroRLE_Rans;
    case XType::SymbolRans: return mgard_x::lossless_type::SymbolRans;
    }
    return mgard_x::lossless_type::Huffman;
}

inline mgard_x::Config MakeConfig(XType t, int level, uint32_t dict, uint32_t block) {
    mgard_x::Config cfg;
    cfg.lossless = ToLossless(t);
    cfg.huff_dict_size = dict;
    cfg.huff_block_size = block;
    cfg.zstd_compress_level = level > 0 ? level : 3;
    // The callers guarantee every code < dict, so MGARD-X's outlier front end never fires; a small ratio
    // keeps its outlier buffers (n * ratio entries) small instead of n-sized.
    cfg.estimate_outlier_ratio = 0.01;
    cfg.log_level = 0;
    return cfg;
}

template <typename Device>
struct Impl {
    using CLC = mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, Device>;
    using SymArray = mgard_x::Array<1, std::int32_t, Device>;
    using ByteArray = mgard_x::Array<1, mgard_x::Byte, Device>;

    // One persistent compressor (MGARD-X workspace + output buffer) per (type, level, dict, block, n),
    // LRU-evicted beyond kMaxEntries. MGARD-X objects are not thread safe: all entry points take Mutex().
    struct Entry { XType t; int level; uint32_t dict, block; mgard_x::SIZE n; uint64_t last_use; std::unique_ptr<CLC> clc; ByteArray comp; };
    static constexpr size_t kMaxEntries = 3;

    static std::mutex& Mutex() { static std::mutex m; return m; }
    static void InitOnce() {
        static std::once_flag once;
        std::call_once(once, []() { mgard_x::DeviceRuntime<Device>::Initialize(); });
    }
    static Entry& Get(XType t, int level, uint32_t dict, uint32_t block, mgard_x::SIZE n) {
        static std::vector<std::unique_ptr<Entry>> cache; static uint64_t tick = 0;
        for (auto& e : cache)
            if (e->t == t && e->level == level && e->dict == dict && e->block == block && e->n == n) { e->last_use = ++tick; return *e; }
        if (cache.size() >= kMaxEntries) {
            size_t victim = 0; for (size_t i = 1; i < cache.size(); ++i) if (cache[i]->last_use < cache[victim]->last_use) victim = i;
            cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(victim));
        }
        auto e = std::make_unique<Entry>();
        e->t = t; e->level = level; e->dict = dict; e->block = block; e->n = n; e->last_use = ++tick;
        e->clc = std::make_unique<CLC>(n, MakeConfig(t, level, dict, block));
        cache.push_back(std::move(e));
        return *cache.back();
    }

    // Compress `arr` (zigzag codes, all < dict; may be modified by MGARD-X's outlier pass) into e.comp and
    // copy the stream to `out`. Caller holds Mutex().
    static bool CompressArray(Entry& e, SymArray& arr, std::vector<uint8_t>& out) {
        e.clc->Compress(arr, e.comp, 0);
        e.clc->Serialize(e.comp, 0);     // required for lossless_type::Huffman (no-op otherwise), as in MGARD-X's Compressor
        mgard_x::DeviceRuntime<Device>::SyncQueue(0);
        if (!DeviceErrorCheck<Device>()) { std::cerr << "[hpmgard xhuff] device error after MGARD-X lossless compress\n"; return false; }
        const size_t bytes = e.comp.shape(0);
        const mgard_x::Byte* p = e.comp.hostCopy(false, 0);
        mgard_x::DeviceRuntime<Device>::SyncQueue(0);
        out.assign(p, p + bytes);
        return true;
    }
    // Decode a stream (device copy in `comp`) into `dec` (n symbols). Caller holds Mutex().
    static bool DecompressArray(Entry& e, ByteArray& comp, SymArray& dec) {
        e.clc->Deserialize(comp, 0);     // inverse of Serialize above
        e.clc->Decompress(comp, dec, 0);
        mgard_x::DeviceRuntime<Device>::SyncQueue(0);
        if (!DeviceErrorCheck<Device>()) { std::cerr << "[hpmgard xhuff] device error after MGARD-X lossless decompress\n"; return false; }
        return true;
    }

    // Host-resident input.
    static bool Encode(const std::vector<int32_t>& codes, XType t, int level, uint32_t dict, uint32_t block, std::vector<uint8_t>& out) {
        InitOnce();
        std::lock_guard<std::mutex> lk(Mutex());
        const mgard_x::SIZE n = static_cast<mgard_x::SIZE>(codes.size());
        try {
            Entry& e = Get(t, level, dict, block, n);
            SymArray arr({n}); arr.load(codes.data());
            if (!CompressArray(e, arr, out)) return false;
            if (VerifyEnabled()) {
                ByteArray comp({static_cast<mgard_x::SIZE>(out.size())}); comp.load(out.data());
                SymArray dec({n});
                if (!DecompressArray(e, comp, dec)) return false;
                const std::int32_t* p = dec.hostCopy(false, 0);
                mgard_x::DeviceRuntime<Device>::SyncQueue(0);
                if (std::memcmp(p, codes.data(), codes.size() * sizeof(int32_t)) != 0) {
                    std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " stream failed the decode check on this device; using the fallback coder\n";
                    return false;
                }
            }
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " compress failed: " << ex.what() << "\n";
            return false;
        }
    }
    static bool Decode(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, XType t, int level, uint32_t dict, uint32_t block) {
        InitOnce();
        std::lock_guard<std::mutex> lk(Mutex());
        try {
            Entry& e = Get(t, level, dict, block, static_cast<mgard_x::SIZE>(n));
            ByteArray comp({static_cast<mgard_x::SIZE>(nb)}); comp.load(reinterpret_cast<const mgard_x::Byte*>(b));
            SymArray dec({static_cast<mgard_x::SIZE>(n)});
            if (!DecompressArray(e, comp, dec)) return false;
            const std::int32_t* p = dec.hostCopy(false, 0);
            mgard_x::DeviceRuntime<Device>::SyncQueue(0);
            codes.assign(p, p + n);
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "[hpmgard xhuff] MGARD-X " << XTypeName(t) << " decompress failed: " << ex.what() << "\n";
            return false;
        }
    }
};

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
