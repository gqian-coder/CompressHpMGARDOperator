// Compression-only pipeline: per component, Gauss L2 restriction to the base order, closed (corner)
// representation, base split (cell average or vertex dedup), quantisation and lossless coding, with the
// residual taken against the DECODED base (closed loop), so the pointwise bound is the residual
// quantiser's alone and the base quantum is a pure rate-allocation knob.
#pragma once
#include "hpmgard/Header.h"
#include "hpmgard/Types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace hpmgard {

struct StreamStat { int component; StreamKind kind; std::string codec; uint64_t count; uint64_t bytes; };

struct CompressStats {
    uint64_t raw_bytes = 0, compressed_bytes = 0, header_bytes = 0;
    std::vector<StreamStat> streams;
    double t_stats = 0, t_refactor = 0, t_base = 0, t_residual = 0, t_code = 0, t_tune = 0, t_total = 0;   // seconds
    double t_geom = 0;   // dedup geometry build (0 when served from the process-wide cache)
    double t_extract = 0, t_upload = 0;   // host-side copies: component extraction (0 on the block-resident GPU path), upload
    CompressConfig used;
    std::string tune_report;
    double ratio() const { return compressed_bytes ? double(raw_bytes) / double(compressed_bytes) : 0.0; }
};

struct CompressResult {
    std::vector<uint8_t> bytes;
    CompressStats stats;
};

// mesh may be null unless cfg.base_method == DedupSfc (then it must hold NumCorners(shape) corners per
// cell in tensor corner order; see CornerPermutation).
CompressResult Compress(const FieldView& field, const CompressConfig& cfg, const MeshCorners* mesh, const BlockMeta& meta);

struct DecompressStats {
    double t_total = 0;      // seconds, whole call
    double t_decode = 0;     // header parse, stream decoding (host zstd, device or host symbol stages) and reconstruction kernels
    double t_download = 0;   // GPU: one chunked, pipelined copy of the interleaved block to the caller's buffer
    double t_insert = 0;     // host-produced components written into the interleaved output (constant, raw, MGARD-X, CPU path)
};

// Writes ncells * nsp * ncomp doubles in the original interleaved layout straight into out (out_count must be
// exactly that many): no intermediate vector, no zero fill, no second copy. mesh is needed only for blocks
// written with DedupSfc; it must be the same mesh (geometry CRCs are checked).
void DecompressInto(const uint8_t* bytes, size_t n, const MeshCorners* mesh, double* out, size_t out_count, BlockHeader* header_out = nullptr, const std::string& device = "auto", DecompressStats* stats = nullptr);
// Same, into a vector (resized when its size differs; a vector of the right size is reused without a zero fill).
void Decompress(const uint8_t* bytes, size_t n, const MeshCorners* mesh, std::vector<double>& out, BlockHeader* header_out = nullptr, const std::string& device = "auto", DecompressStats* stats = nullptr);

BlockHeader ReadHeader(const uint8_t* bytes, size_t n);
std::string DescribeHeader(const BlockHeader& h);

// Field statistics per component (min, max, sum of squares) and the resulting per-component scale.
struct FieldStats { std::vector<double> lo, hi, sumsq; };
FieldStats ComputeStats(const FieldView& f);
std::vector<double> ScaleFor(const FieldStats& s, size_t count_per_comp, Norm norm);

} // namespace hpmgard
