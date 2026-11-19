// Self-describing block header, MGARD-X style: fixed preamble with versions, size and CRC32, then
// fixed-width fields, length-prefixed strings and a stream directory. See documents/bitstream.md.
#pragma once
#include "hpmgard/Types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace hpmgard {

constexpr uint8_t kOperatorTypeByte = 54;
constexpr uint8_t kSoftwareVersion[3] = {0, 1, 0};
constexpr uint8_t kFileVersion[3] = {1, 0, 0};

enum class StreamKind : uint8_t { Avg = 0, Remainder = 1, Residual = 2, Direct = 3, Const = 4, RawDoubles = 5 };
enum class StreamCodec : uint8_t { SymbolChain = 0, MgardX = 1, RawDoubles = 2, Const = 3 };

struct StreamRec {
    uint8_t component = 0;
    StreamKind kind = StreamKind::Residual;
    StreamCodec codec = StreamCodec::SymbolChain;
    uint64_t offset = 0;       // from the start of the payload
    uint64_t bytes = 0;
    uint64_t count = 0;        // number of values represented
};

struct BlockHeader {
    // preamble
    uint8_t software_version[3] = {kSoftwareVersion[0], kSoftwareVersion[1], kSoftwareVersion[2]};
    uint8_t file_version[3] = {kFileVersion[0], kFileVersion[1], kFileVersion[2]};
    uint16_t flags = 0;        // bit0 raw block, bit1 mesh identity, bit2 geometry CRCs, bit3 directory
    // block
    uint64_t blockid = 0;
    std::vector<uint64_t> dims;
    uint8_t data_type = 0;
    // discretisation
    ElemShape shape = ElemShape::HEX;
    uint8_t p = 0;
    uint16_t nsp = 0;
    uint8_t ncomp = 1;
    uint8_t dof_layout = 0;    // 0 = lexicographic, first axis fastest, components interleaved per point
    NodeSet node_set = NodeSet::GaussLegendre;
    // configuration
    uint8_t base_order = 1;
    BaseRepresentation base_rep = BaseRepresentation::Closed;
    BaseMethod base_method = BaseMethod::CellAvg;
    Norm norm = Norm::RelL2;
    double tolerance = 0.0;                 // as requested (relative or absolute)
    std::vector<double> var_scale;          // per component
    std::vector<double> tol_abs;            // per component, pointwise bound of the output
    std::vector<double> base_quantum;       // per component
    std::vector<double> residual_quantum;   // per component
    double amp_inf = 1.0;
    std::string base_codec, residual_codec;
    uint8_t direct_mgardx = 0;
    // mesh identity (flag bit1) and geometry CRCs (bit2)
    std::string conn_var, coords_var;
    uint8_t corner_perm_code = 0;
    uint64_t num_vertex_groups = 0;
    uint32_t crc_vid = 0, crc_perm = 0;
    // provenance
    std::string writer;                     // e.g. "hpmgard 0.1.0 cpu"
    uint8_t mgardx_version[3] = {0, 0, 0};
    uint8_t writer_backend = 0;             // 0 CPU, 1 HIP, 2 CUDA
    // directory (bit3)
    std::vector<StreamRec> streams;

    bool raw_block() const { return flags & 1u; }
};

// Serialises preamble + metadata; the payload is appended by the caller right after.
std::vector<uint8_t> SerializeHeader(const BlockHeader& h);
// Parses the header; returns the number of bytes consumed (= payload offset).
size_t ParseHeader(const uint8_t* b, size_t n, BlockHeader& h);
// Reads only the versions and the operator type (cheap check).
bool PeekVersions(const uint8_t* b, size_t n, uint8_t software[3], uint8_t file[3]);

} // namespace hpmgard
