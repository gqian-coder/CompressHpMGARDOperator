#include "operator/CompressHpMGARDOperator.h"
#include "hpmgard/hpmgard.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace adios2 {
namespace plugin {

namespace {
std::mutex g_mutex;   // the compressor keeps per-process caches (mesh); serialise calls
std::string Upper(std::string s) { for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return s; }
std::string Lower(std::string s) { for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }
}

CompressHpMGARDOperator::CompressHpMGARDOperator(const Params& parameters)
    : PluginOperatorInterface(parameters) {}

void CompressHpMGARDOperator::AddExtraParameters(const Params& params) {
    for (const auto& kv : params) {
        if (kv.first == "EngineName") m_EngineName = kv.second;
        else if (kv.first == "VariableName") m_VariableName = kv.second;
    }
}

std::string CompressHpMGARDOperator::Param(const std::string& name, const std::string& def) const {
    // explicit parameter wins, then HPMGARD_<NAME> from the environment, then the default
    auto it = m_Parameters.find(name);
    if (it != m_Parameters.end() && !it->second.empty()) return it->second;
    const std::string env = "HPMGARD_" + Upper(name);
    if (const char* e = std::getenv(env.c_str())) if (*e) return std::string(e);
    return def;
}

bool CompressHpMGARDOperator::IsDataTypeValid(const DataType type) const {
    return type == DataType::Double || type == DataType::Float;
}

size_t CompressHpMGARDOperator::GetEstimatedSize(const size_t ElemCount, const size_t ElemSize, const size_t, const size_t*) const {
    // raw-double fallback per component plus header and stream containers
    return ElemCount * std::max<size_t>(ElemSize, 8) + 65536;
}

size_t CompressHpMGARDOperator::Operate(const char* dataIn, const Dims& /*blockStart*/, const Dims& blockCount, const DataType type, char* bufferOut) {
    std::lock_guard<std::mutex> lock(g_mutex);
    using namespace hpmgard;
    if (!IsDataTypeValid(type)) throw std::invalid_argument("[CompressHpMGARDOperator] only double and float are supported");
    CompressConfig cfg;
    const std::string cell_type = Param("cell_type");
    if (cell_type.empty()) throw std::invalid_argument("[CompressHpMGARDOperator] missing parameter 'cell_type'");
    cfg.shape = ParseShape(cell_type);
    const std::string order = Param("order");
    if (order.empty()) throw std::invalid_argument("[CompressHpMGARDOperator] missing parameter 'order'");
    cfg.p = std::stoi(order);
    const int nsp = NumSP(cfg.shape, cfg.p);
    size_t total = 1; for (size_t d : blockCount) total *= d;
    const size_t ncells = blockCount.size() >= 1 ? blockCount[0] : 0;
    if (blockCount.size() == 1) {
        // flattened block: cells = total / (nsp * components)
        const int nc = std::stoi(Param("components", "1"));
        if (total % (static_cast<size_t>(nsp) * nc)) throw std::invalid_argument("[CompressHpMGARDOperator] block size is not a multiple of nsp*components");
        cfg.ncomp = nc;
    } else {
        if (total % (ncells * static_cast<size_t>(nsp))) throw std::invalid_argument("[CompressHpMGARDOperator] block dims " + std::to_string(blockCount[0]) + "x" + std::to_string(total / blockCount[0]) + " do not match nsp=" + std::to_string(nsp));
        cfg.ncomp = static_cast<int>(total / (ncells * static_cast<size_t>(nsp)));
        const std::string comp = Param("components");
        if (!comp.empty() && std::stoi(comp) != cfg.ncomp) throw std::invalid_argument("[CompressHpMGARDOperator] 'components' disagrees with the block dims");
    }
    const size_t cells = blockCount.size() == 1 ? total / (static_cast<size_t>(nsp) * cfg.ncomp) : ncells;
    cfg.node_set = Lower(Param("node_set", "gauss")) == "lobatto" ? NodeSet::GaussLobatto : NodeSet::GaussLegendre;
    const std::string tol = Param("tolerance", Param("accuracy"));
    if (tol.empty()) throw std::invalid_argument("[CompressHpMGARDOperator] missing parameter 'tolerance'");
    const std::string mode = Upper(Param("mode", "REL_L2"));
    cfg.norm = mode == "ABS" ? Norm::Abs : mode == "REL_RANGE" || mode == "REL_VAL" ? Norm::RelRange : Norm::RelL2;
    {   // "tolerance" may be a comma-separated per-component list of ABSOLUTE bounds (one bound per variable
        // and component across all blocks, as a driver with a variable-wide pre-pass supplies)
        std::vector<double> list; std::string cur;
        for (char ch : tol) { if (ch == ',') { if (!cur.empty()) list.push_back(std::stod(cur)); cur.clear(); } else cur += ch; }
        if (!cur.empty()) list.push_back(std::stod(cur));
        if (list.empty()) throw std::invalid_argument("[CompressHpMGARDOperator] empty 'tolerance'");
        cfg.tolerance = list[0];
        if (list.size() > 1 || cfg.norm == Norm::Abs) {
            if (list.size() == 1) list.assign(cfg.ncomp, list[0]);
            if (static_cast<int>(list.size()) != cfg.ncomp) throw std::invalid_argument("[CompressHpMGARDOperator] 'tolerance' list must have one entry per component");
            cfg.abs_tolerance = list; cfg.norm = Norm::Abs;
        }
    }
    cfg.tune = Lower(Param("tuning", "auto")) != "fixed";
    cfg.tune_sample = std::stod(Param("tuning_sample", "0.03"));
    cfg.tune_codecs = Param("tune_codecs", cfg.tune_codecs);
    cfg.base_order = std::stoi(Param("base_order", "1"));
    const std::string bm = Lower(Param("base_method", ""));
    const std::string meshfile = Param("meshfile"), conn = Param("connectivity_variable"), coords = Param("coordinates_variable");
    const bool have_mesh = !meshfile.empty() && !conn.empty() && !coords.empty();
    cfg.base_method = bm == "dedup_sfc" || bm == "dedup" ? BaseMethod::DedupSfc : bm == "plain" ? BaseMethod::Plain : bm == "cellavg" ? BaseMethod::CellAvg : (have_mesh ? BaseMethod::DedupSfc : BaseMethod::CellAvg);
    cfg.base_quantum_ratio = std::stod(Param("base_quantum", "0.5"));
    cfg.base_codec = Param("base_codec", mgardx::Available() ? "huffman|zstd:3" : "huffman|zstd:3");
    cfg.residual_codec = Param("residual_codec", "huffman|zstd:3");
    cfg.allow_mgardx_direct = Param("allow_mgardx_direct", "1") != "0";
    cfg.threads = std::stoi(Param("threads", "0"));
    cfg.verbose = Param("verbose", "0") != "0";
    const size_t threshold = std::stoul(Param("threshold", "100000"));

    const size_t blockid = std::stoul(Param("blockid", "0"));
    const size_t raw_bytes = total * (type == DataType::Double ? 8 : 4);
    BlockMeta meta; meta.blockid = blockid; meta.dims.assign(blockCount.begin(), blockCount.end()); meta.data_type = type == DataType::Double ? 0 : 1;
    if (blockCount.size() == 1) meta.dims = {cells, static_cast<uint64_t>(nsp) * cfg.ncomp};
    meta.conn_var = conn; meta.coords_var = coords;

    // below the threshold: header with the raw flag + raw bytes
    if (raw_bytes < threshold) {
        BlockHeader h; h.flags = 1; h.blockid = blockid; h.dims = meta.dims; h.data_type = meta.data_type; h.shape = cfg.shape; h.p = static_cast<uint8_t>(cfg.p);
        h.nsp = static_cast<uint16_t>(nsp); h.ncomp = static_cast<uint8_t>(cfg.ncomp); h.node_set = cfg.node_set; h.norm = cfg.norm; h.tolerance = cfg.tolerance;
        h.var_scale.assign(cfg.ncomp, 0); h.tol_abs.assign(cfg.ncomp, 0); h.base_quantum.assign(cfg.ncomp, 0); h.residual_quantum.assign(cfg.ncomp, 0);
        h.writer = "hpmgard raw";
        const std::vector<uint8_t> hb = SerializeHeader(h);
        std::memcpy(bufferOut, hb.data(), hb.size()); std::memcpy(bufferOut + hb.size(), dataIn, raw_bytes);
        return hb.size() + raw_bytes;
    }
    // double view of the block
    std::vector<double> converted;
    const double* src = reinterpret_cast<const double*>(dataIn);
    if (type == DataType::Float) { const float* f = reinterpret_cast<const float*>(dataIn); converted.assign(f, f + total); src = converted.data(); }
    FieldView field{src, cells, nsp, cfg.ncomp};
    const MeshCorners* mesh = nullptr;
    if (cfg.base_method == BaseMethod::DedupSfc && have_mesh) {
        try { mesh = &m_Mesh.Load(meshfile, conn, coords, blockid, cfg.shape); }
        catch (const std::exception& e) {
            // e.g. the mesh is being written in the same step: fall back to the cell-average base
            std::cerr << "[CompressHpMGARDOperator] WARNING: mesh not readable for dedup (" << e.what() << "); using cellavg for " << m_VariableName << " block " << blockid << "\n";
            cfg.base_method = BaseMethod::CellAvg; meta.conn_var.clear(); meta.coords_var.clear();
        }
    }
    if (cfg.base_method != BaseMethod::DedupSfc) { meta.conn_var.clear(); meta.coords_var.clear(); }
    CompressResult r = Compress(field, cfg, mesh, meta);
    if (r.bytes.size() >= raw_bytes + 1024) {
        // incompressible: store raw
        BlockHeader h = ReadHeader(r.bytes.data(), r.bytes.size()); h.flags = 1; h.streams.clear();
        const std::vector<uint8_t> hb = SerializeHeader(h);
        std::memcpy(bufferOut, hb.data(), hb.size()); std::memcpy(bufferOut + hb.size(), dataIn, raw_bytes);
        return hb.size() + raw_bytes;
    }
    if (cfg.verbose) std::cerr << "[CompressHpMGARDOperator] " << m_VariableName << " block " << blockid << ": CR " << r.stats.ratio() << " in " << r.stats.t_total << " s\n" << r.stats.tune_report;
    std::memcpy(bufferOut, r.bytes.data(), r.bytes.size());
    return r.bytes.size();
}

size_t CompressHpMGARDOperator::InverseOperate(const char* bufferIn, const size_t sizeIn, char* dataOut) {
    std::lock_guard<std::mutex> lock(g_mutex);
    using namespace hpmgard;
    const uint8_t* in = reinterpret_cast<const uint8_t*>(bufferIn);
    BlockHeader h;
    const size_t off = ParseHeader(in, sizeIn, h);
    size_t total = 1; for (uint64_t d : h.dims) total *= d;
    const size_t out_bytes = total * (h.data_type ? 4 : 8);
    if (h.raw_block()) { std::memcpy(dataOut, in + off, out_bytes); return out_bytes; }
    const MeshCorners* mesh = nullptr;
    if (h.base_method == BaseMethod::DedupSfc) {
        // explicit mesh (params or env) must come as the full triple; otherwise the file being read + header names
        std::string mf = Param("meshfile"), cv = Param("connectivity_variable"), xv = Param("coordinates_variable");
        const int given = (!mf.empty()) + (!cv.empty()) + (!xv.empty());
        if (given != 0 && given != 3) throw std::invalid_argument("[CompressHpMGARDOperator] meshfile, connectivity_variable and coordinates_variable must be given together (or none, to read the mesh from the compressed file)");
        if (given == 0) {
            if (h.conn_var.empty() || h.coords_var.empty()) throw std::runtime_error("[CompressHpMGARDOperator] this block was written with dedup_sfc but names no mesh; set HPMGARD_MESHFILE, HPMGARD_CONNECTIVITY_VARIABLE and HPMGARD_COORDINATES_VARIABLE");
            if (m_EngineName.empty()) throw std::runtime_error("[CompressHpMGARDOperator] engine name unavailable; set HPMGARD_MESHFILE (+ variable names) explicitly");
            mf = m_EngineName; cv = h.conn_var; xv = h.coords_var;
        }
        mesh = &m_Mesh.Load(mf, cv, xv, h.blockid, h.shape);
    }
    std::vector<double> out;
    Decompress(in, sizeIn, mesh, out);
    if (out.size() != total) throw std::runtime_error("[CompressHpMGARDOperator] decoded size mismatch");
    if (h.data_type) { float* f = reinterpret_cast<float*>(dataOut); for (size_t i = 0; i < total; ++i) f[i] = static_cast<float>(out[i]); }
    else std::memcpy(dataOut, out.data(), out_bytes);
    return out_bytes;
}

} // namespace plugin
} // namespace adios2

extern "C" {
adios2::plugin::CompressHpMGARDOperator* OperatorCreate(const adios2::Params& parameters) { return new adios2::plugin::CompressHpMGARDOperator(parameters); }
void OperatorDestroy(adios2::plugin::CompressHpMGARDOperator* obj) { delete obj; }
}
