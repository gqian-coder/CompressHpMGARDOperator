#include "hpmgard/Compressor.h"
#include "hpmgard/Backend.h"
#include "hpmgard/BaseSplit.h"
#include "hpmgard/Bytes.h"
#include "hpmgard/MgardX.h"
#include "hpmgard/Operators.h"
#include "hpmgard/Tuner.h"
#include "hpmgard/codec/Codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdexcept>

namespace hpmgard {

namespace {

using Clock = std::chrono::steady_clock;
inline double Sec(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Gather one component of an interleaved block into a contiguous array.
void ExtractComponent(const FieldView& f, int c, std::vector<double>& out) {
    const size_t npts = f.ncells * static_cast<size_t>(f.nsp);
    out.resize(npts);
    if (f.ncomp == 1) { std::memcpy(out.data(), f.data, npts * sizeof(double)); return; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npts; ++i) out[i] = f.data[i * f.ncomp + c];
}

void InsertComponent(const std::vector<double>& in, int c, int ncomp, double* out) {
    const size_t npts = in.size();
    if (ncomp == 1) { std::memcpy(out, in.data(), npts * sizeof(double)); return; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npts; ++i) out[i * ncomp + c] = in[i];
}

struct Payload {
    std::vector<uint8_t> bytes;
    std::vector<StreamRec> dir;
    void add(int comp, StreamKind kind, StreamCodec codec, uint64_t count, const std::vector<uint8_t>& b) {
        StreamRec r; r.component = static_cast<uint8_t>(comp); r.kind = kind; r.codec = codec; r.offset = bytes.size(); r.bytes = b.size(); r.count = count;
        dir.push_back(r); bytes.insert(bytes.end(), b.begin(), b.end());
    }
};

std::vector<uint8_t> DoublesToBytes(const std::vector<double>& v) {
    std::vector<uint8_t> b(v.size() * 8);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

} // namespace

FieldStats ComputeStats(const FieldView& f) {
    FieldStats s;
    const int nc = f.ncomp;
    s.lo.assign(nc, std::numeric_limits<double>::infinity()); s.hi.assign(nc, -std::numeric_limits<double>::infinity()); s.sumsq.assign(nc, 0.0);
    const size_t npts = f.ncells * static_cast<size_t>(f.nsp);
    for (int c = 0; c < nc; ++c) {
        double lo = s.lo[c], hi = s.hi[c], ss = 0.0;
#pragma omp parallel for schedule(static) reduction(min : lo) reduction(max : hi) reduction(+ : ss)
        for (size_t i = 0; i < npts; ++i) { const double v = f.data[i * nc + c]; lo = std::min(lo, v); hi = std::max(hi, v); ss += v * v; }
        s.lo[c] = lo; s.hi[c] = hi; s.sumsq[c] = ss;
    }
    return s;
}

std::vector<double> ScaleFor(const FieldStats& s, size_t count_per_comp, Norm norm) {
    std::vector<double> sc(s.lo.size(), 1.0);
    for (size_t c = 0; c < sc.size(); ++c) {
        if (norm == Norm::RelL2) sc[c] = count_per_comp ? std::sqrt(s.sumsq[c] / double(count_per_comp)) : 0.0;
        else if (norm == Norm::RelRange) sc[c] = s.hi[c] - s.lo[c];
    }
    return sc;
}

BlockHeader ReadHeader(const uint8_t* bytes, size_t n) { BlockHeader h; ParseHeader(bytes, n, h); return h; }

std::string DescribeHeader(const BlockHeader& h) {
    std::ostringstream o;
    o << "hpMGARD block v" << int(h.file_version[0]) << "." << int(h.file_version[1]) << "." << int(h.file_version[2])
      << " (software " << int(h.software_version[0]) << "." << int(h.software_version[1]) << "." << int(h.software_version[2]) << ", writer '" << h.writer << "')\n"
      << "  block " << h.blockid << " dims"; for (auto d : h.dims) o << " " << d; o << (h.data_type ? " float" : " double") << (h.raw_block() ? " RAW" : "") << "\n"
      << "  " << ShapeName(h.shape) << " p=" << int(h.p) << " nsp=" << h.nsp << " ncomp=" << int(h.ncomp) << " nodes=" << (h.node_set == NodeSet::GaussLegendre ? "gauss-legendre" : "gauss-lobatto") << "\n"
      << "  base order " << int(h.base_order) << " rep=" << (h.base_rep == BaseRepresentation::Closed ? "closed" : "data-nodes")
      << " method=" << (h.base_method == BaseMethod::CellAvg ? "cellavg" : h.base_method == BaseMethod::DedupSfc ? "dedup_sfc" : "plain")
      << " amp_inf=" << h.amp_inf << " base_codec='" << h.base_codec << "' residual_codec='" << h.residual_codec << "'" << (h.direct_mgardx ? " DIRECT-MGARDX" : "") << "\n"
      << "  norm=" << (h.norm == Norm::Abs ? "abs" : h.norm == Norm::RelL2 ? "rel_l2" : "rel_range") << " tolerance=" << h.tolerance << "\n";
    for (int c = 0; c < h.ncomp; ++c) o << "  comp " << c << ": scale=" << h.var_scale[c] << " tol_abs=" << h.tol_abs[c] << " q_base=" << h.base_quantum[c] << " q_res=" << h.residual_quantum[c] << "\n";
    if (h.flags & 2u) o << "  mesh: conn='" << h.conn_var << "' coords='" << h.coords_var << "' groups=" << h.num_vertex_groups << " crc_vid=" << h.crc_vid << " crc_perm=" << h.crc_perm << "\n";
    for (const StreamRec& s : h.streams) {
        static const char* kinds[] = {"avg", "remainder", "residual", "direct", "const", "raw"};
        static const char* codecs[] = {"chain", "mgardx", "raw", "const"};
        o << "  stream comp=" << int(s.component) << " " << kinds[static_cast<int>(s.kind)] << " " << codecs[static_cast<int>(s.codec)] << " count=" << s.count << " bytes=" << s.bytes << "\n";
    }
    return o.str();
}

CompressResult Compress(const FieldView& field, const CompressConfig& cfg_in, const MeshCorners* mesh, const BlockMeta& meta) {
    const auto t0 = Clock::now();
    if (cfg_in.threads > 0) omp_set_num_threads(cfg_in.threads);
    CompressConfig cfg = cfg_in;
    if (field.nsp != NumSP(cfg.shape, cfg.p)) throw std::invalid_argument("Compress: nsp does not match (shape, p)");
    if (field.ncomp != cfg.ncomp) throw std::invalid_argument("Compress: ncomp mismatch");
    if (cfg.base_order < 0 || cfg.base_order > cfg.p) throw std::invalid_argument("Compress: base_order must be in [0, p]");
    if (!cfg.abs_tolerance.empty() && cfg.abs_tolerance.size() != static_cast<size_t>(cfg.ncomp)) throw std::invalid_argument("Compress: abs_tolerance must have ncomp entries");
    const size_t npts = field.ncells * static_cast<size_t>(field.nsp);
    const uint64_t raw_bytes = field.size() * sizeof(double);

    CompressResult res;
    CompressStats& st = res.stats;
    st.raw_bytes = raw_bytes;

    // --- tuner --------------------------------------------------------------------------------------
    if (cfg.tune) {
        const auto tt = Clock::now();
        TuneReport rep;
        cfg = Tune(field, cfg, mesh, &rep);
        cfg.tune = false;
        st.t_tune = Sec(tt, Clock::now());
        st.tune_report = rep.text;
    }
    if (cfg.base_method == BaseMethod::DedupSfc) {
        const bool ok = mesh && !mesh->empty() && mesh->ncells() == field.ncells && mesh->ncorner == NumCorners(cfg.shape)
                        && cfg.base_order == 1 && IsTensorShape(cfg.shape);
        if (!ok) cfg.base_method = BaseMethod::CellAvg;   // dedup needs corner geometry and a p1 closed base
    }
    if (cfg.direct_mgardx && !mgardx::Available()) cfg.direct_mgardx = false;
    std::unique_ptr<Backend> be = MakeBackend(cfg.device);
    st.used = cfg;

    // --- statistics ---------------------------------------------------------------------------------
    const auto ts = Clock::now();
    const FieldStats fs = ComputeStats(field);
    const std::vector<double> scale = ScaleFor(fs, npts, cfg.norm);
    st.t_stats = Sec(ts, Clock::now());

    BlockHeader h;
    h.blockid = meta.blockid; h.dims = meta.dims; h.data_type = meta.data_type;
    if (h.dims.empty()) h.dims = {field.ncells, static_cast<uint64_t>(field.nsp) * field.ncomp};
    else {
        uint64_t prod = 1; for (uint64_t d : h.dims) prod *= d;
        if (h.dims[0] != field.ncells || prod != field.size())
            throw std::invalid_argument("Compress: BlockMeta.dims do not match the field (dims[0] must be the cell count and the product the value count)");
    }
    h.shape = cfg.shape; h.p = static_cast<uint8_t>(cfg.p); h.nsp = static_cast<uint16_t>(field.nsp); h.ncomp = static_cast<uint8_t>(field.ncomp);
    h.node_set = cfg.node_set; h.base_order = static_cast<uint8_t>(cfg.base_order); h.base_method = cfg.base_method; h.norm = cfg.abs_tolerance.empty() ? cfg.norm : Norm::Abs; h.tolerance = cfg.abs_tolerance.empty() ? cfg.tolerance : cfg.abs_tolerance[0];
    h.base_codec = cfg.base_codec; h.residual_codec = cfg.residual_codec; h.direct_mgardx = cfg.direct_mgardx ? 1 : 0;
    h.conn_var = meta.conn_var; h.coords_var = meta.coords_var;
    h.var_scale = scale; h.tol_abs.assign(field.ncomp, 0.0); h.base_quantum.assign(field.ncomp, 0.0); h.residual_quantum.assign(field.ncomp, 0.0);
    { int ma, mi, pa; mgardx::Version(ma, mi, pa); h.mgardx_version[0] = ma; h.mgardx_version[1] = mi; h.mgardx_version[2] = pa; }
    h.writer = std::string("hpmgard ") + std::to_string(kSoftwareVersion[0]) + "." + std::to_string(kSoftwareVersion[1]) + "." + std::to_string(kSoftwareVersion[2]) + " " + be->name();
    h.writer_backend = static_cast<uint8_t>(be->type());

    OperatorSet ops;
    const bool refactor = !cfg.direct_mgardx;
    if (refactor) { ops = BuildOperators(cfg.shape, cfg.p, cfg.base_order, cfg.node_set, cfg.base_rep); h.base_rep = ops.base_rep; h.amp_inf = ops.amp_inf; }
    DedupGeometry geom;
    if (refactor && cfg.base_method == BaseMethod::DedupSfc) {
        geom = BuildDedupGeometry(*mesh);
        h.num_vertex_groups = geom.ngroups; h.crc_vid = geom.crc_vid; h.crc_perm = geom.crc_perm; h.corner_perm_code = 1;
    }
    const std::vector<codec::Stage> base_chain = (cfg.base_codec == "mgardx") ? std::vector<codec::Stage>{} : codec::ParseChain(cfg.base_codec);
    const bool base_mgardx = (cfg.base_codec == "mgardx");
    if (base_mgardx && !mgardx::Available()) throw std::invalid_argument("base_codec=mgardx but hpMGARD was built without MGARD-X");
    const std::vector<codec::Stage> res_chain = codec::ParseChain(cfg.residual_codec);

    Payload pay;
    std::vector<double> u, base, base_dec, ub, resid, avg, rem;
    std::vector<int32_t> k;
    for (int c = 0; c < field.ncomp; ++c) {
        const auto tc0 = Clock::now();
        ExtractComponent(field, c, u);
        const double tol = !cfg.abs_tolerance.empty() ? cfg.abs_tolerance[c] : (cfg.norm == Norm::Abs) ? cfg.tolerance : cfg.tolerance * scale[c];
        h.tol_abs[c] = tol;
        // constant component: one value
        if (fs.hi[c] == fs.lo[c]) {
            std::vector<double> one{fs.lo[c]};
            pay.add(c, StreamKind::Const, StreamCodec::Const, npts, DoublesToBytes(one));
            st.streams.push_back({c, StreamKind::Const, "const", npts, 8});
            continue;
        }
        if (!(tol > 0.0) || !std::isfinite(tol)) {
            pay.add(c, StreamKind::RawDoubles, StreamCodec::RawDoubles, npts, DoublesToBytes(u));
            st.streams.push_back({c, StreamKind::RawDoubles, "raw", npts, npts * 8});
            continue;
        }
        const double qres = 2.0 * tol;
        h.residual_quantum[c] = qres;
        if (!refactor) {
            const auto tm = Clock::now();
            std::vector<uint8_t> b = mgardx::Compress1D(u.data(), npts, tol, std::numeric_limits<double>::infinity(), 0);
            st.t_code += Sec(tm, Clock::now());
            pay.add(c, StreamKind::Direct, StreamCodec::MgardX, npts, b);
            st.streams.push_back({c, StreamKind::Direct, "mgardx", npts, b.size()});
            continue;
        }
        // --- base ---
        const auto tr = Clock::now();
        const size_t nbase = field.ncells * static_cast<size_t>(ops.nsp_base);
        base.resize(nbase);
        be->Restrict(ops, u.data(), field.ncells, base.data());
        st.t_refactor += Sec(tr, Clock::now());
        const auto tb = Clock::now();
        const double qb = 2.0 * cfg.base_quantum_ratio * tol;
        h.base_quantum[c] = qb;
        base_dec.assign(nbase, 0.0);
        const int nb = ops.nsp_base;
        auto code_symbols = [&](const std::vector<int32_t>& syms, StreamKind kind, const std::vector<codec::Stage>& chain, double q) {
            const auto tk = Clock::now();
            std::vector<uint8_t> b = codec::EncodeSymbols(syms.data(), syms.size(), chain, q);
            st.t_code += Sec(tk, Clock::now());
            pay.add(c, kind, StreamCodec::SymbolChain, syms.size(), b);
            st.streams.push_back({c, kind, codec::ChainString(chain), syms.size(), b.size()});
        };
        auto code_doubles_mgardx = [&](const std::vector<double>& vals, StreamKind kind, double abs_tol, std::vector<double>& dec) {
            const auto tk = Clock::now();
            std::vector<uint8_t> b = mgardx::Compress1D(vals.data(), vals.size(), abs_tol, std::numeric_limits<double>::infinity(), 0);
            mgardx::Decompress(b.data(), b.size(), dec);
            st.t_code += Sec(tk, Clock::now());
            pay.add(c, kind, StreamCodec::MgardX, vals.size(), b);
            st.streams.push_back({c, kind, "mgardx", vals.size(), b.size()});
        };
        bool base_ok = true;
        if (cfg.base_method == BaseMethod::Plain) {
            if (base_mgardx) { code_doubles_mgardx(base, StreamKind::Avg, qb / 2.0, base_dec); }
            else {
                k.resize(base.size()); base_ok = be->Quantize(base.data(), base.size(), qb, k.data());
                if (base_ok) { code_symbols(k, StreamKind::Avg, base_chain, qb); for (size_t i = 0; i < nbase; ++i) base_dec[i] = k[i] * qb; }
            }
        } else if (cfg.base_method == BaseMethod::CellAvg) {
            avg.resize(field.ncells);
            be->CellMeans(base.data(), field.ncells, nb, avg.data());
            std::vector<double> avg_dec;
            if (base_mgardx) code_doubles_mgardx(avg, StreamKind::Avg, qb / 2.0, avg_dec);
            else {
                k.resize(avg.size()); base_ok = be->Quantize(avg.data(), avg.size(), qb, k.data());
                if (base_ok) { code_symbols(k, StreamKind::Avg, base_chain, qb); avg_dec.resize(field.ncells); for (size_t i = 0; i < field.ncells; ++i) avg_dec[i] = k[i] * qb; }
            }
            if (base_ok && nb > 1) {
                rem.resize(nbase);
#pragma omp parallel for schedule(static)
                for (size_t cell = 0; cell < field.ncells; ++cell) for (int j = 0; j < nb; ++j) rem[cell * nb + j] = base[cell * nb + j] - avg_dec[cell];
                k.resize(rem.size()); base_ok = be->Quantize(rem.data(), rem.size(), qb, k.data());
                if (base_ok) {
                    code_symbols(k, StreamKind::Remainder, res_chain, qb);
#pragma omp parallel for schedule(static)
                    for (size_t cell = 0; cell < field.ncells; ++cell) for (int j = 0; j < nb; ++j) base_dec[cell * nb + j] = avg_dec[cell] + k[cell * nb + j] * qb;
                }
            } else if (base_ok) {
                for (size_t i = 0; i < nbase; ++i) base_dec[i] = avg_dec[i];
            }
        } else {   // DedupSfc: per-vertex mean + Hilbert-ordered remainder
            const size_t ndof = nbase;   // = ncells * 8
            avg.assign(geom.ngroups, 0.0);
            std::vector<uint32_t> cnt(geom.ngroups, 0);
            for (size_t i = 0; i < ndof; ++i) { avg[geom.vid[i]] += base[i]; ++cnt[geom.vid[i]]; }
            for (uint32_t g = 0; g < geom.ngroups; ++g) if (cnt[g]) avg[g] /= cnt[g];
            std::vector<double> avg_dec;
            if (base_mgardx) code_doubles_mgardx(avg, StreamKind::Avg, qb / 2.0, avg_dec);
            else {
                k.resize(avg.size()); base_ok = be->Quantize(avg.data(), avg.size(), qb, k.data());
                if (base_ok) { code_symbols(k, StreamKind::Avg, base_chain, qb); avg_dec.resize(geom.ngroups); for (size_t i = 0; i < geom.ngroups; ++i) avg_dec[i] = k[i] * qb; }
            }
            if (base_ok) {
                rem.resize(ndof);
#pragma omp parallel for schedule(static)
                for (size_t slot = 0; slot < ndof; ++slot) { const uint32_t d = geom.perm[slot]; rem[slot] = base[d] - avg_dec[geom.vid[d]]; }
                k.resize(rem.size()); base_ok = be->Quantize(rem.data(), rem.size(), qb, k.data());
                if (base_ok) {
                    code_symbols(k, StreamKind::Remainder, res_chain, qb);
#pragma omp parallel for schedule(static)
                    for (size_t slot = 0; slot < ndof; ++slot) { const uint32_t d = geom.perm[slot]; base_dec[d] = avg_dec[geom.vid[d]] + k[slot] * qb; }
                }
            }
        }
        if (!base_ok) {
            // base quantiser overflow: store the component raw (keeps the bound trivially)
            pay.dir.erase(std::remove_if(pay.dir.begin(), pay.dir.end(), [&](const StreamRec& r) { return r.component == c; }), pay.dir.end());
            pay.add(c, StreamKind::RawDoubles, StreamCodec::RawDoubles, npts, DoublesToBytes(u));
            st.streams.push_back({c, StreamKind::RawDoubles, "raw", npts, npts * 8});
            continue;
        }
        st.t_base += Sec(tb, Clock::now());
        // --- residual against the decoded base (closed loop) ---
        const auto tq = Clock::now();
        ub.resize(npts);
        be->Prolong(ops, base_dec.data(), field.ncells, ub.data());
        k.resize(npts);
        const bool res_ok = be->ResidualQuantize(u.data(), ub.data(), npts, qres, k.data());
        st.t_residual += Sec(tq, Clock::now());
        if (!res_ok) {
            pay.dir.erase(std::remove_if(pay.dir.begin(), pay.dir.end(), [&](const StreamRec& r) { return r.component == c; }), pay.dir.end());
            pay.add(c, StreamKind::RawDoubles, StreamCodec::RawDoubles, npts, DoublesToBytes(u));
            st.streams.push_back({c, StreamKind::RawDoubles, "raw", npts, npts * 8});
            continue;
        }
        code_symbols(k, StreamKind::Residual, res_chain, qres);
        (void)tc0;
    }
    // rebuild the stream list in payload order (raw fallbacks may have reordered the record list)
    h.streams = pay.dir;
    std::vector<uint8_t> header = SerializeHeader(h);
    if (header.size() + pay.bytes.size() >= raw_bytes + header.size()) { /* never larger than raw + header by construction of the fallbacks */ }
    res.bytes = std::move(header);
    res.bytes.insert(res.bytes.end(), pay.bytes.begin(), pay.bytes.end());
    st.header_bytes = res.bytes.size() - pay.bytes.size();
    st.compressed_bytes = res.bytes.size();
    st.t_total = Sec(t0, Clock::now());
    if (cfg.verbose) {
        std::cerr << "[hpmgard] block " << meta.blockid << ": " << raw_bytes << " -> " << st.compressed_bytes << " bytes (CR " << st.ratio() << ") in " << st.t_total << " s"
                  << " [stats " << st.t_stats << " refactor " << st.t_refactor << " base " << st.t_base << " residual " << st.t_residual << " code " << st.t_code << " tune " << st.t_tune << "]\n";
        for (const auto& s : st.streams) std::cerr << "    comp " << s.component << " kind " << static_cast<int>(s.kind) << " " << s.codec << " " << s.count << " values -> " << s.bytes << " bytes\n";
    }
    return res;
}

void Decompress(const uint8_t* bytes, size_t n, const MeshCorners* mesh, std::vector<double>& out, BlockHeader* header_out, const std::string& device) {
    BlockHeader h;
    const size_t off = ParseHeader(bytes, n, h);
    if (header_out) *header_out = h;
    const size_t ncells = h.dims.empty() ? 0 : h.dims[0];
    const size_t npts = ncells * h.nsp;
    out.assign(npts * h.ncomp, 0.0);
    const uint8_t* payload = bytes + off;
    const size_t payload_bytes = n - off;
    OperatorSet ops; bool have_ops = false;
    DedupGeometry geom; bool have_geom = false;
    std::unique_ptr<Backend> be = MakeBackend(device);
    std::vector<double> comp(npts), base_dec, avg_dec; std::vector<int32_t> k;
    for (int c = 0; c < h.ncomp; ++c) {
        // collect this component's streams
        const StreamRec *s_avg = nullptr, *s_rem = nullptr, *s_res = nullptr, *s_other = nullptr;
        for (const StreamRec& s : h.streams) {
            if (s.component != c) continue;
            if (s.offset + s.bytes > payload_bytes) throw std::runtime_error("Decompress: stream exceeds payload");
            if (s.kind == StreamKind::Avg) s_avg = &s; else if (s.kind == StreamKind::Remainder) s_rem = &s; else if (s.kind == StreamKind::Residual) s_res = &s; else s_other = &s;
        }
        if (s_other) {
            if (s_other->kind == StreamKind::Const) { double v; std::memcpy(&v, payload + s_other->offset, 8); std::fill(comp.begin(), comp.end(), v); }
            else if (s_other->kind == StreamKind::RawDoubles) { std::memcpy(comp.data(), payload + s_other->offset, npts * 8); }
            else if (s_other->kind == StreamKind::Direct) { std::vector<double> d; mgardx::Decompress(payload + s_other->offset, s_other->bytes, d); if (d.size() != npts) throw std::runtime_error("Decompress: MGARD-X size mismatch"); comp.swap(d); comp.resize(npts); }
            else throw std::runtime_error("Decompress: unknown stream kind");
            InsertComponent(comp, c, h.ncomp, out.data());
            continue;
        }
        if (!s_res) throw std::runtime_error("Decompress: residual stream missing");
        if (!have_ops) { ops = BuildOperators(h.shape, h.p, h.base_order, h.node_set, h.base_rep); have_ops = true; }
        const int nb = ops.nsp_base;
        const size_t nbase = ncells * static_cast<size_t>(nb);
        base_dec.assign(nbase, 0.0);
        const double qb = h.base_quantum[c];
        auto decode_doubles = [&](const StreamRec& s, std::vector<double>& vals) {
            if (s.codec == StreamCodec::MgardX) { mgardx::Decompress(payload + s.offset, s.bytes, vals); if (vals.size() != s.count) throw std::runtime_error("Decompress: MGARD-X count mismatch"); }
            else { codec::DecodeSymbols(payload + s.offset, s.bytes, k); if (k.size() != s.count) throw std::runtime_error("Decompress: symbol count mismatch"); vals.resize(k.size()); for (size_t i = 0; i < k.size(); ++i) vals[i] = k[i] * qb; }
        };
        if (h.base_method == BaseMethod::Plain) {
            if (!s_avg) throw std::runtime_error("Decompress: base stream missing");
            decode_doubles(*s_avg, base_dec);
        } else if (h.base_method == BaseMethod::CellAvg) {
            if (!s_avg) throw std::runtime_error("Decompress: average stream missing");
            decode_doubles(*s_avg, avg_dec);
            if (s_rem) {
                codec::DecodeSymbols(payload + s_rem->offset, s_rem->bytes, k);
                if (k.size() != nbase) throw std::runtime_error("Decompress: remainder count mismatch");
#pragma omp parallel for schedule(static)
                for (size_t cell = 0; cell < ncells; ++cell) for (int j = 0; j < nb; ++j) base_dec[cell * nb + j] = avg_dec[cell] + k[cell * nb + j] * qb;
            } else {
                for (size_t cell = 0; cell < ncells; ++cell) for (int j = 0; j < nb; ++j) base_dec[cell * nb + j] = avg_dec[cell];
            }
        } else {
            if (!mesh || mesh->empty()) throw std::runtime_error("Decompress: this block was written with dedup_sfc and needs the mesh corners");
            if (!have_geom) {
                geom = BuildDedupGeometry(*mesh); have_geom = true;
                if (geom.ngroups != h.num_vertex_groups || geom.crc_vid != h.crc_vid || geom.crc_perm != h.crc_perm)
                    throw std::runtime_error("Decompress: mesh geometry does not match the block (vertex groups or Hilbert order differ)");
            }
            if (!s_avg || !s_rem) throw std::runtime_error("Decompress: dedup streams missing");
            decode_doubles(*s_avg, avg_dec);
            codec::DecodeSymbols(payload + s_rem->offset, s_rem->bytes, k);
            if (k.size() != nbase) throw std::runtime_error("Decompress: remainder count mismatch");
#pragma omp parallel for schedule(static)
            for (size_t slot = 0; slot < nbase; ++slot) { const uint32_t d = geom.perm[slot]; base_dec[d] = avg_dec[geom.vid[d]] + k[slot] * qb; }
        }
        be->Prolong(ops, base_dec.data(), ncells, comp.data());
        codec::DecodeSymbols(payload + s_res->offset, s_res->bytes, k);
        if (k.size() != npts) throw std::runtime_error("Decompress: residual count mismatch");
        be->DequantizeAdd(k.data(), npts, h.residual_quantum[c], comp.data());
        InsertComponent(comp, c, h.ncomp, out.data());
    }
}

} // namespace hpmgard
