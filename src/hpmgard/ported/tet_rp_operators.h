/**
 * @file tet_rp_operators.h
 * @brief Self-contained Restriction / Prolongation operators for tetrahedron elements.
 *
 * TET elements use full dense R/P matrices extracted from hpMusic's
 * SchemeCoefGSpGFpTet. Unlike prism (Kronecker of TRI × 1D) or hex
 * (tensor-product), TET operators have no decomposition — they are
 * stored as full dense matrices.
 *
 * TET SP count: n_tet(p) = (p+1)(p+2)(p+3)/6
 *
 * This module has NO runtime dependency on hpMusic.
 */

#ifndef PMGPU_TET_RP_OPERATORS_H
#define PMGPU_TET_RP_OPERATORS_H

#include "rp_operators.h"       // DenseMatrix, RPOperatorSet
#include <vector>
#include <string>
#include <array>

namespace pmgpu {

static constexpr int kMaxTetOrder = 5;

/**
 * @brief Builds and caches R/P operator tables for TET elements.
 */
class TetRPOperatorTable {
public:
    TetRPOperatorTable() = default;

    /**
     * @brief Build all TET operators up to p_max.
     * @param p_max Maximum polynomial order (<= kMaxTetOrder = 5).
     */
    void Build(int p_max);

    /// @name Accessors
    /// @{
    const DenseMatrix& GetProlongFull(int p)   const { return prolong_full_[p]; }
    const DenseMatrix& GetRestrictFull(int p)  const { return restrict_full_[p]; }

    int  MaxOrder() const { return max_order_; }
    bool IsBuilt()  const { return built_; }
    /// @}

    /// @name CPU Apply (interleaved multi-variable layout)
    /// @{
    void ApplyRestrict(int p_hi,
                       const double* u_hi, int n_sp_hi,
                       double* u_lo, int& n_sp_lo,
                       int nv) const;

    void ApplyProlong(int p_lo,
                      const double* u_lo, int n_sp_lo,
                      double* u_hi, int& n_sp_hi,
                      int nv) const;
    /// @}

    /**
     * @brief Verify R·P = I for all levels. Returns max deviation.
     */
    double VerifyRP() const;
    double VerifyRP(int p) const;

    void PrintSummary() const;

private:
    int max_order_ = 0;
    bool built_ = false;

    // Full TET operators: prolong_full_[p] maps p -> p+1, restrict_full_[p] maps p -> p-1
    std::array<DenseMatrix, kMaxTetOrder + 1> prolong_full_;   // [p] maps p -> p+1
    std::array<DenseMatrix, kMaxTetOrder + 1> restrict_full_;  // [p] maps p -> p-1

    // Helpers
    void LoadProlongMatrices();
    void LoadRestrictMatrices();
};

} // namespace pmgpu

#endif // PMGPU_TET_RP_OPERATORS_H
