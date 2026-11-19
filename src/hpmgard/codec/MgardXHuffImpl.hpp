// Template shared by the SERIAL and HIP translation units of the MGARD-X Huffman stage.
// Included only from those two files (each compiled for its own MGARD-X backend).
#pragma once
#include "hpmgard/codec/MgardXHuff.h"
#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

namespace hpmgard {
namespace codec {
namespace xhuff {

template <typename Device>
struct Impl {
    static void InitOnce() {
        static std::once_flag once;
        std::call_once(once, []() { mgard_x::DeviceRuntime<Device>::Initialize(); });
    }
    static mgard_x::Config MakeConfig(int zstd_level) {
        mgard_x::Config cfg;
        cfg.lossless = zstd_level > 0 ? mgard_x::lossless_type::Huffman_Zstd : mgard_x::lossless_type::Huffman;
        cfg.huff_dict_size = kDictSize;
        cfg.huff_block_size = kBlockSize;
        cfg.zstd_compress_level = zstd_level > 0 ? zstd_level : 3;
        cfg.estimate_outlier_ratio = 1.0;
        cfg.log_level = 0;
        return cfg;
    }
    // codes: zigzag-mapped, all < kDictSize (checked by the caller)
    static bool Encode(const std::vector<int32_t>& codes, int zstd_level, std::vector<uint8_t>& out) {
        InitOnce();
        const mgard_x::Config cfg = MakeConfig(zstd_level);
        const mgard_x::SIZE n = static_cast<mgard_x::SIZE>(codes.size());
        try {
            mgard_x::Array<1, std::int32_t, Device> arr({n});
            arr.load(codes.data());
            mgard_x::Array<1, mgard_x::Byte, Device> comp;
            mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, Device> clc(n, cfg);
            clc.Compress(arr, comp, 0);
            clc.Serialize(comp, 0);     // required for lossless_type::Huffman (no-op for Huffman_Zstd), as in MGARD-X's Compressor
            mgard_x::DeviceRuntime<Device>::SyncQueue(0);
            const size_t bytes = comp.shape(0);
            const mgard_x::Byte* p = comp.hostCopy(false, 0);
            mgard_x::DeviceRuntime<Device>::SyncQueue(0);
            out.assign(p, p + bytes);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[hpmgard xhuff] MGARD-X lossless compress failed: " << e.what() << "\n";
            return false;
        }
    }
    static bool Decode(const uint8_t* b, size_t nb, std::vector<int32_t>& codes, size_t n, int zstd_level) {
        InitOnce();
        const mgard_x::Config cfg = MakeConfig(zstd_level);
        try {
            mgard_x::Array<1, mgard_x::Byte, Device> comp({static_cast<mgard_x::SIZE>(nb)});
            comp.load(reinterpret_cast<const mgard_x::Byte*>(b));
            mgard_x::Array<1, std::int32_t, Device> dec({static_cast<mgard_x::SIZE>(n)});
            mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, Device> clc(static_cast<mgard_x::SIZE>(n), cfg);
            clc.Deserialize(comp, 0);   // inverse of Serialize above
            clc.Decompress(comp, dec, 0);
            mgard_x::DeviceRuntime<Device>::SyncQueue(0);
            const std::int32_t* p = dec.hostCopy(false, 0);
            mgard_x::DeviceRuntime<Device>::SyncQueue(0);
            codes.assign(p, p + n);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[hpmgard xhuff] MGARD-X lossless decompress failed: " << e.what() << "\n";
            return false;
        }
    }
};

} // namespace xhuff
} // namespace codec
} // namespace hpmgard
