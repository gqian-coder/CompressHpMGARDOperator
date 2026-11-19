// Core types of the hpMGARD compression library (compression-only v1).
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpmgard {

// Values match mp_mgard::ElemShape so metadata stays comparable across the two projects.
enum class ElemShape : uint8_t { QUAD = 3, TET = 4, HEX = 5, PRISM = 6, PYRAMID = 7 };

inline int ShapeDim(ElemShape s) { return s == ElemShape::QUAD ? 2 : 3; }
inline bool IsTensorShape(ElemShape s) { return s == ElemShape::QUAD || s == ElemShape::HEX; }

inline int NumSP(ElemShape s, int p) {
    switch (s) {
    case ElemShape::QUAD:    return (p + 1) * (p + 1);
    case ElemShape::HEX:     return (p + 1) * (p + 1) * (p + 1);
    case ElemShape::TET:     return (p + 1) * (p + 2) * (p + 3) / 6;
    case ElemShape::PRISM:   return (p + 1) * (p + 1) * (p + 2) / 2;
    case ElemShape::PYRAMID: return (p + 1) * (p + 2) * (2 * p + 3) / 6;
    }
    throw std::invalid_argument("NumSP: unknown shape");
}

inline int NumCorners(ElemShape s) {
    switch (s) {
    case ElemShape::QUAD: return 4;
    case ElemShape::HEX: return 8;
    case ElemShape::TET: return 4;
    case ElemShape::PRISM: return 6;
    case ElemShape::PYRAMID: return 5;
    }
    throw std::invalid_argument("NumCorners: unknown shape");
}

const char* ShapeName(ElemShape s);
ElemShape ParseShape(const std::string& name);   // "hexahedron", "hex", "vtkDGHex", ...

enum class NodeSet : uint8_t { GaussLegendre = 0, GaussLobatto = 1 };
enum class Norm : uint8_t { Abs = 0, RelL2 = 1, RelRange = 2 };
// How the base level is split before coding.
enum class BaseMethod : uint8_t { Plain = 0, CellAvg = 1, DedupSfc = 2 };
// Where the base values live: at the data node set of order p_base, or at the closed (Gauss-Lobatto)
// node set of order p_base (p_base = 1: the cell corners), so that the prolongation interpolates.
enum class BaseRepresentation : uint8_t { DataNodes = 0, Closed = 1 };

// A block of a DG field: data[cell * nsp * ncomp + sp * ncomp + comp].
struct FieldView {
    const double* data = nullptr;
    size_t ncells = 0;
    int nsp = 0;
    int ncomp = 1;
    size_t size() const { return ncells * static_cast<size_t>(nsp) * static_cast<size_t>(ncomp); }
};

// Corner geometry of the cells (only needed for BaseMethod::DedupSfc):
// vertex_id[cell * ncorner + k] = mesh vertex id of corner k in tensor order (i fastest),
// xyz[3 * (cell * ncorner + k) + d] = its coordinate.
struct MeshCorners {
    int ncorner = 0;
    std::vector<int64_t> vertex_id;
    std::vector<double> xyz;
    bool empty() const { return vertex_id.empty(); }
    size_t ncells() const { return ncorner ? vertex_id.size() / static_cast<size_t>(ncorner) : 0; }
};

struct CompressConfig {
    ElemShape shape = ElemShape::HEX;
    int p = 0;               // order of the data
    int ncomp = 1;
    NodeSet node_set = NodeSet::GaussLegendre;
    Norm norm = Norm::RelL2;
    double tolerance = 1e-3; // relative (RelL2, RelRange) or absolute (Abs); the pointwise bound of the output
    // Optional per-component absolute pointwise bounds (size ncomp). When set they override tolerance/norm,
    // e.g. a driver passing one variable-wide bound to every block of a multi-block variable.
    std::vector<double> abs_tolerance;
    // --- hierarchy ---
    int base_order = 1;                      // 0 = cell mean, 1 = trilinear, ..., p = no refactoring
    BaseRepresentation base_rep = BaseRepresentation::Closed;
    BaseMethod base_method = BaseMethod::CellAvg;
    // The base is coded in a closed loop (the residual is taken against the decoded base), so its
    // quantum does not enter the error bound; it only trades base bytes against residual bytes.
    double base_quantum_ratio = 0.5;         // base quantum = 2 * ratio * tol_abs
    // --- codecs (chain specs, e.g. "huffman|zstd:3", "frbp", "raw|zstd:19", "mgardx") ---
    std::string base_codec = "huffman|zstd:3";
    std::string residual_codec = "huffman|zstd:3";
    bool direct_mgardx = false;              // compress the whole component with MGARD-X 1-D instead
    // --- tuner ---
    bool tune = false;
    double tune_sample = 0.03;
    std::string tune_codecs = "huffman|zstd:3,raw|zstd:3,frbp|zstd:3";
    bool allow_mgardx_direct = true;
    // --- runtime ---
    std::string device = "auto";          // auto | cpu | hip | cuda (backend for the numeric stages)
    int threads = 0;
    bool verbose = false;
};

// Identity of the block, written into the header.
struct BlockMeta {
    uint64_t blockid = 0;
    std::vector<uint64_t> dims;      // ADIOS block dims, e.g. {ncells, nsp*ncomp}
    uint8_t data_type = 0;           // 0 double, 1 float (original type of the caller's buffer)
    std::string conn_var;            // mesh identity for self-contained decodes (DedupSfc)
    std::string coords_var;
};

} // namespace hpmgard
