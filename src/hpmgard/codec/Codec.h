// Lossless coding building blocks for quantised integer streams, chainable for testing:
//   stage 0 (symbols -> bytes): raw | frbp | huffman[:dict_bits] | xhuff[:zstd_level] | xrle | xrans
//     (xhuff = MGARD-X Huffman[+zstd inside MGARD-X], xrle = MGARD-X ZeroRLE_Rans, xrans = MGARD-X SymbolRans; GPU-capable)
//   later stages (bytes -> bytes): zstd[:level]
// A chain is written as a spec string, e.g. "huffman|zstd:3", "frbp", "raw|zstd:19", "frbp|zstd:3".
// EncodeSymbols produces a self-describing container, so DecodeSymbols needs no parameters.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hpmgard {
namespace codec {

enum class StageId : uint8_t { Raw = 0, FRBP = 1, Huffman = 2, Zstd = 3, MgardXHuffman = 4, MgardXZeroRleRans = 5, MgardXSymbolRans = 6 };
inline bool IsMgardXStage(StageId id) { return id == StageId::MgardXHuffman || id == StageId::MgardXZeroRleRans || id == StageId::MgardXSymbolRans; }

struct Stage {
    StageId id = StageId::Raw;
    int param = 0;            // huffman: dict_bits (default 16); zstd: level (default 3); xhuff: zstd level inside MGARD-X (0 = Huffman only)
};

std::vector<Stage> ParseChain(const std::string& spec);
std::string ChainString(const std::vector<Stage>& chain);
const char* StageName(StageId id);

// --- individual stages -------------------------------------------------------------------------
std::vector<uint8_t> RawEncode(const int32_t* sym, size_t n);
void RawDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n);

// Fixed-rate bit packing in 32-value blocks (mp-MGARD FRBP format, FIXED mode): 40-byte header,
// one rate byte per block (padded to 4), then per non-zero block a sign word and `rate` plane words.
std::vector<uint8_t> FrbpEncode(const int32_t* sym, size_t n, double eb_info);
void FrbpDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n);

// Canonical Huffman over zigzag-mapped symbols with a 2^dict_bits dictionary and escaped outliers.
std::vector<uint8_t> HuffmanEncode(const int32_t* sym, size_t n, int dict_bits);
void HuffmanDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n);

std::vector<uint8_t> ZstdEncode(const uint8_t* b, size_t n, int level);
void ZstdDecode(const uint8_t* b, size_t nb, uint8_t* out, size_t n);

// --- chains ------------------------------------------------------------------------------------
// device: 0 CPU, 1 HIP, 2 CUDA (used by the MGARD-X stages; others ignore it). d_sym: optional device
// pointer to the same symbols; an MGARD-X first stage then codes straight from device memory (no host
// copy) and only falls back to `sym` when that is not possible.
std::vector<uint8_t> EncodeSymbols(const int32_t* sym, size_t n, const std::vector<Stage>& chain, double eb_info = 0.0, int device = 0, const int32_t* d_sym = nullptr);
// Loads kernels / allocates the MGARD-X workspace for a device-capable first stage (no-op otherwise).
void WarmupStage(const Stage& first, int device);
// Device the MGARD-X stages actually use for `device` (0 when MGARD-X is unavailable or fails its preflight there).
int MgardXDevice(int device);
// Default chains: "xhuff|zstd:3" where the MGARD-X stages run on a GPU, "huffman|zstd:3" otherwise.
std::string DefaultChain(int device);
std::string DefaultTuneCodecs(int device);
// Device-resident symbols: codes straight from device memory when the first stage can, otherwise fetches the
// host copy through `host()` (called at most once) and codes it like EncodeSymbols.
std::vector<uint8_t> EncodeSymbolsDevice(const int32_t* d_sym, size_t n, const std::vector<Stage>& chain, double eb_info, int device, const std::function<const int32_t*()>& host);
// Decodes a stream onto the device when its first stage can (MGARD-X stage, GPU device); returns the device
// pointer (valid until the next codec call) and the symbol count, or nullptr when the host path must be used.
const int32_t* DecodeSymbolsToDevice(const uint8_t* b, size_t nb, size_t& n_out, int device);
// Returns the number of symbols decoded (== the count written by EncodeSymbols).
size_t DecodeSymbols(const uint8_t* b, size_t nb, std::vector<int32_t>& out, int device = 0);
// Whether a first stage can run (e.g. xhuff needs MGARD-X, and a HIP device for device 1).
bool StageAvailable(StageId id, int device = 0);
size_t PeekSymbolCount(const uint8_t* b, size_t nb);

} // namespace codec
} // namespace hpmgard
