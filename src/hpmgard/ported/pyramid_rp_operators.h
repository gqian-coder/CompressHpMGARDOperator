/**
 * @file pyramid_rp_operators.h
 * @brief Self-contained Restriction / Prolongation operators for pyramid elements.
 *
 * PYRAMID elements use full dense R/P matrices extracted from hpMusic's
 * SchemeCoefGSpGFpPyramid. Like TET, these have no tensor-product
 * decomposition — they are stored as full dense matrices.
 *
 * PYRAMID SP count: n_pyramid(p) = (p+1)(p+2)(2p+3)/6
 *
 * This module has NO runtime dependency on hpMusic.
 */

#ifndef PMGPU_PYRAMID_RP_OPERATORS_H
#define PMGPU_PYRAMID_RP_OPERATORS_H

#include "rp_operators.h"       // DenseMatrix, RPOperatorSet
#include <vector>
#include <string>
#include <array>

namespace pmgpu {

static constexpr int kMaxPyramidOrder = 5;

/**
 * @brief Builds and caches R/P operator tables for PYRAMID elements.
 */
class PyramidRPOperatorTable {
public:
    PyramidRPOperatorTable() = default;

    /**
     * @brief Build all PYRAMID operators up to p_max.
     * @param p_max Maximum polynomial order (<= kMaxPyramidOrder = 5).
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

    // Full PYRAMID operators
    std::array<DenseMatrix, kMaxPyramidOrder + 1> prolong_full_;   // [p] maps p -> p+1
    std::array<DenseMatrix, kMaxPyramidOrder + 1> restrict_full_;  // [p] maps p -> p-1

    // Helpers
    void LoadProlongMatrices();
    void LoadRestrictMatrices();
};

} // namespace pmgpu

#endif // PMGPU_PYRAMID_RP_OPERATORS_H
