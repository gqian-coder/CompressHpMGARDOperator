#include "operator/MeshAdapter.h"
#include "hpmgard/BaseSplit.h"
#include <adios2.h>
#include <stdexcept>
#include <vector>

namespace hpmgard {
namespace plugin {

namespace {
// A single ADIOS instance for all mesh reads of this process (as the centroid plugin does).
adios2::ADIOS& Adios() { static adios2::ADIOS ad; return ad; }
int g_io_counter = 0;
}

void MeshAdapter::Clear() { key_.clear(); corners_ = MeshCorners{}; }

const MeshCorners& MeshAdapter::Load(const std::string& meshfile, const std::string& conn_var, const std::string& coords_var,
                                     size_t blockid, ElemShape shape) {
    const std::string key = meshfile + "|" + conn_var + "|" + coords_var + "|" + std::to_string(blockid) + "|" + std::to_string(static_cast<int>(shape));
    if (key == key_ && !corners_.empty()) return corners_;
    adios2::IO io = Adios().DeclareIO("hpmgard_mesh_" + std::to_string(g_io_counter++));
    adios2::Engine eng = io.Open(meshfile, adios2::Mode::ReadRandomAccess);
    auto vconn = io.InquireVariable<int64_t>(conn_var);
    if (!vconn) throw std::runtime_error("MeshAdapter: connectivity variable '" + conn_var + "' not found in " + meshfile);
    const auto cblocks = eng.BlocksInfo(vconn, 0);
    if (blockid >= cblocks.size()) throw std::runtime_error("MeshAdapter: block " + std::to_string(blockid) + " out of range for " + conn_var);
    const adios2::Dims ccount = cblocks[blockid].Count;
    if (ccount.size() != 2) throw std::runtime_error("MeshAdapter: connectivity must be {ncells, nodes_per_cell}");
    const size_t ncells = ccount[0], npc = ccount[1];
    std::vector<int64_t> conn(ncells * npc);
    if (vconn.Shape().empty()) vconn.SetBlockSelection(blockid);
    else vconn.SetSelection({cblocks[blockid].Start, ccount});
    eng.Get(vconn, conn.data(), adios2::Mode::Sync);

    std::vector<double> pts;
    const std::string ctype = io.VariableType(coords_var);
    if (ctype == "double") {
        auto v = io.InquireVariable<double>(coords_var);
        const auto pb = eng.BlocksInfo(v, 0);
        if (blockid >= pb.size()) throw std::runtime_error("MeshAdapter: coordinate block out of range");
        size_t n = 1; for (auto c : pb[blockid].Count) n *= c;
        pts.resize(n);
        if (v.Shape().empty()) v.SetBlockSelection(blockid); else v.SetSelection({pb[blockid].Start, pb[blockid].Count});
        eng.Get(v, pts.data(), adios2::Mode::Sync);
    } else if (ctype == "float") {
        auto v = io.InquireVariable<float>(coords_var);
        const auto pb = eng.BlocksInfo(v, 0);
        if (blockid >= pb.size()) throw std::runtime_error("MeshAdapter: coordinate block out of range");
        size_t n = 1; for (auto c : pb[blockid].Count) n *= c;
        std::vector<float> f(n);
        if (v.Shape().empty()) v.SetBlockSelection(blockid); else v.SetSelection({pb[blockid].Start, pb[blockid].Count});
        eng.Get(v, f.data(), adios2::Mode::Sync);
        pts.assign(f.begin(), f.end());
    } else throw std::runtime_error("MeshAdapter: coordinate variable '" + coords_var + "' must be double or float");
    eng.Close();
    const size_t npts = pts.size() / 3;

    const int nc = NumCorners(shape);
    const std::vector<int> perm = CornerPermutation(shape);
    if (npc < static_cast<size_t>(nc)) throw std::runtime_error("MeshAdapter: fewer connectivity nodes than cell corners");
    MeshCorners m; m.ncorner = nc; m.vertex_id.resize(ncells * nc); m.xyz.resize(3 * ncells * nc);
    for (size_t c = 0; c < ncells; ++c)
        for (int k = 0; k < nc; ++k) {
            const int64_t v = conn[c * npc + perm[k]];
            if (v < 0 || static_cast<size_t>(v) >= npts) throw std::runtime_error("MeshAdapter: connectivity index out of range (block-local ids expected)");
            m.vertex_id[c * nc + k] = v;
            for (int d = 0; d < 3; ++d) m.xyz[3 * (c * nc + k) + d] = pts[3 * v + d];
        }
    corners_ = std::move(m); key_ = key;
    return corners_;
}

} // namespace plugin
} // namespace hpmgard
