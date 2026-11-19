/**
 * @file sfc.h
 * @brief Space-filling-curve ordering of points by coordinate.
 *
 * Used by BaseMethod::DEDUP_SFC to order the p1 residual DOFs so that each
 * 32-value FRBP block covers one small neighbourhood and can share a rate.
 *
 * The permutation is derived from mesh geometry alone, so it is reproducible at
 * decompress time from the same coordinates and need not be stored in the
 * bitstream (the same accounting CompressMGARDCentroidOperator uses).
 *
 * Scale-invariant: coordinates are quantised against their own bounding box, so
 * a mesh file in different units than the solution still yields the same order.
 */

#ifndef MP_MGARD_CORE_SFC_H
#define MP_MGARD_CORE_SFC_H

#include <algorithm>
#include <cstdint>
#include <vector>

// Hilbert3D is used from both host code and device kernels (sfc_gpu.hip).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define MPM_SFC_HD __host__ __device__
#else
#define MPM_SFC_HD
#endif

namespace mp_mgard {
namespace sfc {

/// 3-D Hilbert index of a point on a 21-bit-per-axis lattice.
MPM_SFC_HD inline uint64_t Hilbert3D(uint32_t x, uint32_t y, uint32_t z) {
    constexpr int b = 21;
    uint32_t X[3] = {x & 0x1fffffu, y & 0x1fffffu, z & 0x1fffffu};
    const uint32_t M = 1u << (b - 1);
    uint32_t P, Q, t;
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (int i = 0; i < 3; ++i) {
            if (X[i] & Q) { X[0] ^= P; }
            else { t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    for (int i = 1; i < 3; ++i) X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1) if (X[2] & Q) t ^= Q - 1;
    for (int i = 0; i < 3; ++i) X[i] ^= t;
    uint64_t h = 0;
    for (int bit = b - 1; bit >= 0; --bit) {
        h = (h << 1) | ((X[0] >> bit) & 1u);
        h = (h << 1) | ((X[1] >> bit) & 1u);
        h = (h << 1) | ((X[2] >> bit) & 1u);
    }
    return h;
}

/// Quantise one coordinate against the bounding box onto the 21-bit lattice.
/// Shared by the host and GPU paths so both produce identical keys.
MPM_SFC_HD inline uint32_t QuantAxis(double v, double lo, double sc) {
    constexpr uint32_t kMaxQ = (1u << 21) - 1u;
    double t = (v - lo) * sc;
    if (t < 0.0) t = 0.0;
    if (t > double(kMaxQ)) t = double(kMaxQ);
    return static_cast<uint32_t>(t);
}

/**
 * @brief Hilbert ordering of `n` cells from interleaved xyz centroids.
 * @param centroid  [3*n] as x0,y0,z0, x1,y1,z1, ...
 * @return perm[i] = index of the original cell occupying slot i.
 *
 * Ties (coincident centroids) fall back to the original index, so the result is
 * deterministic and identical on the compress and decompress sides.
 */
inline std::vector<uint32_t> BuildCellPerm(const double* centroid, size_t n) {
    std::vector<uint32_t> perm(n);
    if (n == 0) return perm;

    double lo[3] = {centroid[0], centroid[1], centroid[2]};
    double hi[3] = {centroid[0], centroid[1], centroid[2]};
    for (size_t c = 1; c < n; ++c)
        for (int d = 0; d < 3; ++d) {
            const double v = centroid[3 * c + d];
            lo[d] = std::min(lo[d], v);
            hi[d] = std::max(hi[d], v);
        }
    const uint32_t maxQ = (1u << 21) - 1u;
    double sc[3];
    for (int d = 0; d < 3; ++d)
        sc[d] = (hi[d] > lo[d]) ? double(maxQ) / (hi[d] - lo[d]) : 0.0;

    std::vector<std::pair<uint64_t, uint32_t>> kv(n);
    for (size_t c = 0; c < n; ++c) {
        const uint32_t qx = QuantAxis(centroid[3 * c + 0], lo[0], sc[0]);
        const uint32_t qy = QuantAxis(centroid[3 * c + 1], lo[1], sc[1]);
        const uint32_t qz = QuantAxis(centroid[3 * c + 2], lo[2], sc[2]);
        kv[c] = {Hilbert3D(qx, qy, qz), static_cast<uint32_t>(c)};
    }
    std::sort(kv.begin(), kv.end(),
              [](const std::pair<uint64_t, uint32_t>& a,
                 const std::pair<uint64_t, uint32_t>& b) {
                  return (a.first != b.first) ? (a.first < b.first)
                                              : (a.second < b.second);
              });
    for (size_t i = 0; i < n; ++i) perm[i] = kv[i].second;
    return perm;
}

} // namespace sfc
} // namespace mp_mgard

#endif // MP_MGARD_CORE_SFC_H
