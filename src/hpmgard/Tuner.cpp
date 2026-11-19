#include "hpmgard/Tuner.h"
#include "hpmgard/Backend.h"
#include "hpmgard/Compressor.h"
#include "hpmgard/MgardX.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace hpmgard {

namespace {

struct Sample {
    std::vector<double> data; FieldView view; MeshCorners mesh; bool has_mesh = false;
};

Sample MakeSample(const FieldView& f, const MeshCorners* mesh, double frac) {
    // Contiguous chunks spread over the block: cells are usually written in a spatially coherent
    // order, so chunks keep most of the vertex sharing that dedup relies on (a strided sample would not).
    Sample s;
    const size_t cell_vals = static_cast<size_t>(f.nsp) * f.ncomp;
    const bool with_mesh = mesh && !mesh->empty() && mesh->ncells() == f.ncells;
    size_t want = std::max<size_t>(2000, static_cast<size_t>(std::ceil(frac * f.ncells)));
    std::vector<std::pair<size_t, size_t>> runs;   // [start, end)
    if (want >= f.ncells) runs.push_back({0, f.ncells});
    else {
        const size_t nchunks = std::min<size_t>(8, std::max<size_t>(1, want / 256));
        const size_t len = (want + nchunks - 1) / nchunks;
        for (size_t k = 0; k < nchunks; ++k) {
            const size_t start = nchunks > 1 ? (k * (f.ncells - len)) / (nchunks - 1) : 0;
            runs.push_back({start, std::min(f.ncells, start + len)});
        }
    }
    size_t n = 0; for (auto& r : runs) n += r.second - r.first;
    s.data.resize(n * cell_vals);
    if (with_mesh) { s.mesh.ncorner = mesh->ncorner; s.mesh.vertex_id.reserve(n * mesh->ncorner); s.mesh.xyz.reserve(3 * n * mesh->ncorner); s.has_mesh = true; }
    size_t k = 0;
    for (auto& r : runs)
        for (size_t c = r.first; c < r.second; ++c, ++k) {
            std::copy(f.data + c * cell_vals, f.data + (c + 1) * cell_vals, s.data.begin() + k * cell_vals);
            if (with_mesh)
                for (int j = 0; j < mesh->ncorner; ++j) { s.mesh.vertex_id.push_back(mesh->vertex_id[c * mesh->ncorner + j]); for (int d = 0; d < 3; ++d) s.mesh.xyz.push_back(mesh->xyz[3 * (c * mesh->ncorner + j) + d]); }
        }
    s.view.data = s.data.data(); s.view.ncells = n; s.view.nsp = f.nsp; s.view.ncomp = f.ncomp;
    return s;
}

std::vector<std::string> SplitList(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

} // namespace

CompressConfig Tune(const FieldView& field, const CompressConfig& cfg_in, const MeshCorners* mesh, TuneReport* report) {
    CompressConfig cfg = cfg_in; cfg.tune = false; cfg.verbose = false;
    const Sample smp = MakeSample(field, mesh, cfg.tune_sample);
    const MeshCorners* smesh = smp.has_mesh ? &smp.mesh : nullptr;
    BlockMeta meta;
    std::ostringstream rep;
    rep << "tuner: sample " << smp.view.ncells << " of " << field.ncells << " cells\n";
    struct Cand { CompressConfig cfg; uint64_t bytes; std::string label; };
    std::vector<Cand> cands;
    auto eval = [&](CompressConfig c, const std::string& label) {
        c.tune = false; c.direct_mgardx = false;
        try {
            CompressResult r = Compress(smp.view, c, smesh, meta);
            cands.push_back({c, r.stats.compressed_bytes, label});
            rep << "  " << label << ": " << r.stats.compressed_bytes << " bytes\n";
        } catch (const std::exception& e) { rep << "  " << label << ": failed (" << e.what() << ")\n"; }
    };
    const std::vector<std::string> codecs = SplitList(cfg.tune_codecs);
    const std::string codec0 = codecs.empty() ? cfg.residual_codec : codecs[0];
    // pass 1: hierarchy and base split at two base quanta, first codec
    std::vector<int> orders{0, 1};
    if (cfg.p >= 3) orders.push_back(2);
    if (cfg.p >= 1 && std::find(orders.begin(), orders.end(), cfg.p) == orders.end()) orders.push_back(cfg.p);   // no split
    std::vector<BaseMethod> methods{BaseMethod::CellAvg};
    if (smesh) methods.push_back(BaseMethod::DedupSfc);
    for (int bo : orders)
        for (BaseMethod bm : methods) {
            if (bm == BaseMethod::DedupSfc && bo != 1) continue;
            if (bo == cfg.p && bm != BaseMethod::CellAvg) continue;
            for (double ratio : {0.5, 2.0}) {
                CompressConfig c = cfg; c.base_order = bo; c.base_method = (bo == 0 || bo == cfg.p) ? BaseMethod::Plain : bm; c.base_quantum_ratio = ratio;
                c.base_codec = codec0; c.residual_codec = codec0;
                std::ostringstream l; l << "p_base=" << bo << " " << (c.base_method == BaseMethod::DedupSfc ? "dedup" : c.base_method == BaseMethod::CellAvg ? "cellavg" : "plain") << " ratio=" << ratio << " " << codec0;
                eval(c, l.str());
            }
        }
    if (cands.empty()) { if (report) report->text = rep.str() + "tuner: all candidates failed, keeping the input configuration\n"; return cfg; }
    auto best_it = std::min_element(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.bytes < b.bytes; });
    CompressConfig best = best_it->cfg;
    // pass 2: refine the base quantum around the winner
    if (best.base_order < cfg.p) {
        for (double f : {0.5, 2.0}) {
            CompressConfig c = best; c.base_quantum_ratio = best.base_quantum_ratio * f;
            std::ostringstream l; l << "p_base=" << c.base_order << " ratio=" << c.base_quantum_ratio << " " << codec0;
            eval(c, l.str());
        }
        best_it = std::min_element(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.bytes < b.bytes; });
        best = best_it->cfg;
    }
    // pass 3: codecs
    for (size_t i = 1; i < codecs.size(); ++i) {
        CompressConfig c = best; c.base_codec = codecs[i]; c.residual_codec = codecs[i];
        eval(c, "codec " + codecs[i]);
    }
    const bool mgardx_ok = mgardx::Available() && mgardx::DeviceWorks(static_cast<int>(MakeBackend(cfg.device)->type()));
    if (!mgardx_ok && mgardx::Available()) rep << "  mgardx candidates skipped: MGARD-X does not work on this device\n";
    if (mgardx_ok) {
        CompressConfig c = best; c.base_codec = "mgardx";
        eval(c, "base codec mgardx");
    }
    best_it = std::min_element(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.bytes < b.bytes; });
    best = best_it->cfg;
    uint64_t best_bytes = best_it->bytes;
    // safety net: direct MGARD-X on the whole sample
    if (cfg.allow_mgardx_direct && mgardx_ok) {
        CompressConfig c = best; c.direct_mgardx = true;
        try {
            CompressResult r = Compress(smp.view, c, smesh, meta);
            rep << "  direct mgardx: " << r.stats.compressed_bytes << " bytes\n";
            if (r.stats.compressed_bytes < best_bytes) { best = c; best_bytes = r.stats.compressed_bytes; }
        } catch (const std::exception& e) { rep << "  direct mgardx: failed (" << e.what() << ")\n"; }
    }
    rep << "tuner: chosen p_base=" << best.base_order << " method=" << (best.base_method == BaseMethod::DedupSfc ? "dedup" : best.base_method == BaseMethod::CellAvg ? "cellavg" : "plain")
        << " ratio=" << best.base_quantum_ratio << " base_codec=" << best.base_codec << " residual_codec=" << best.residual_codec << (best.direct_mgardx ? " DIRECT-MGARDX" : "")
        << " (" << best_bytes << " bytes on the sample)\n";
    if (report) { report->text = rep.str(); report->best_bytes_per_value = double(best_bytes) / double(smp.view.size()); }
    best.tune = false;
    return best;
}

} // namespace hpmgard
