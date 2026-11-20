// Geometry needed by BaseMethod::DedupSfc: dense vertex-group ids of every corner DOF and the
// Hilbert order of the corner DOFs. Both are derived from the mesh (exact vertex ids, integer-quantised
// coordinates), so they are rebuilt identically at decompress time and are protected by CRCs.
#pragma once
#include "hpmgard/Types.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace hpmgard {

struct DedupGeometry {
    uint32_t ngroups = 0;
    std::vector<uint32_t> vid;    // [ncells * ncorner] group of each corner DOF (tensor corner order)
    std::vector<uint32_t> perm;   // [ncells * ncorner] perm[slot] = DOF stored at slot (Hilbert order)
    uint32_t crc_vid = 0, crc_perm = 0;
};

DedupGeometry BuildDedupGeometry(const MeshCorners& mesh);
// Process-wide cache of BuildDedupGeometry keyed by a fingerprint of the mesh arrays (sizes + chunked CRCs),
// so steps and blocks that share a mesh build the Hilbert order and vertex groups once. LRU, capped at
// HPMGARD_GEOM_CACHE_MB (default 1024). Thread safe.
std::shared_ptr<const DedupGeometry> CachedDedupGeometry(const MeshCorners& mesh);
// Seconds spent in BuildDedupGeometry by the last CachedDedupGeometry call of this thread (0 on a cache hit).
double LastGeometryBuildSeconds();

// Permutation from a cell type's native corner order to the tensor corner order (i fastest) used by the
// closed base of order 1. VTK/hpMusic hexahedron corners 0..7 -> tensor slots.
std::vector<int> CornerPermutation(ElemShape shape);

} // namespace hpmgard
