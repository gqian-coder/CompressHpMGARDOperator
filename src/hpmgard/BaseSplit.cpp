#include "hpmgard/BaseSplit.h"
#include "hpmgard/Bytes.h"
#include "ported/sfc.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace hpmgard {

std::vector<int> CornerPermutation(ElemShape shape) {
    switch (shape) {
    // VTK hexahedron corners: 0 (-,-,-) 1 (+,-,-) 2 (+,+,-) 3 (-,+,-) 4..7 the same at +z.
    // Tensor slot (i + 2j + 4k): slot 0 <- 0, 1 <- 1, 2 <- 3, 3 <- 2, 4 <- 4, 5 <- 5, 6 <- 7, 7 <- 6.
    case ElemShape::HEX: return {0, 1, 3, 2, 4, 5, 7, 6};
    case ElemShape::QUAD: return {0, 1, 3, 2};
    default: { std::vector<int> id(NumCorners(shape)); for (size_t i = 0; i < id.size(); ++i) id[i] = static_cast<int>(i); return id; }
    }
}

DedupGeometry BuildDedupGeometry(const MeshCorners& mesh) {
    DedupGeometry g;
    const size_t n = mesh.vertex_id.size();
    if (n == 0 || mesh.xyz.size() != 3 * n) throw std::invalid_argument("BuildDedupGeometry: inconsistent mesh corners");
    // dense group ids in increasing vertex-id order (deterministic)
    std::vector<int64_t> ids(mesh.vertex_id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    g.ngroups = static_cast<uint32_t>(ids.size());
    g.vid.resize(n);
    for (size_t i = 0; i < n; ++i)
        g.vid[i] = static_cast<uint32_t>(std::lower_bound(ids.begin(), ids.end(), mesh.vertex_id[i]) - ids.begin());
    // Hilbert order of the corner DOFs (ties broken by index inside BuildCellPerm)
    g.perm = mp_mgard::sfc::BuildCellPerm(mesh.xyz.data(), n);
    g.crc_vid = Crc32(reinterpret_cast<const uint8_t*>(g.vid.data()), g.vid.size() * 4);
    g.crc_perm = Crc32(reinterpret_cast<const uint8_t*>(g.perm.data()), g.perm.size() * 4);
    return g;
}

} // namespace hpmgard
