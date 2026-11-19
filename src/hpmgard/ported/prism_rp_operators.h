/**
 * @file prism_rp_operators.h
 * @brief Restriction / Prolongation operators for prism elements,
 *        extracted from hpMusic SchemeCoefGSpGFpPrism.
 *
 * The PRISM R/P operators are dense matrices precomputed and stored
 * directly from hpMusic data (NOT Kronecker products of 1D and TRI).
 *
 * Prism SP layout: index = i_zeta * n_tri + i_tri
 * Prism SP count:  n_prism(p) = (p+1) * (p+1)(p+2)/2
 *
 * This module has NO runtime dependency on hpMusic.
 */

#ifndef PMGPU_PRISM_RP_OPERATORS_H
#define PMGPU_PRISM_RP_OPERATORS_H

#include "rp_operators.h"       // DenseMatrix, RPOperatorSet
#include "tri_quadrature.h"     // kMaxTriOrder
#include <vector>
#include <string>

namespace pmgpu {

/**
 * @brief Builds and caches R/P operator tables for prism elements.
 *
 * Uses the same interface as RPOperatorTable (ApplyRestrict, ApplyProlong,
 * VerifyRP, SaveBinary, LoadBinary) but constructs operators specifically
 * for prism elements.
 *
 * Usage:
 * @code
 *   PrismRPOperatorTable table;
 *   table.Build(3);  // up to p=3
 *
 *   table.ApplyRestrict(p_hi, u_hi, n_sp_hi, u_lo, n_sp_lo, nv);
 *   table.ApplyProlong (p_lo, u_lo, n_sp_lo, u_hi, n_sp_hi, nv);
 * @endcode
 */
class PrismRPOperatorTable {
public:
    PrismRPOperatorTable() = default;

    /**
     * @brief Build all prism operators up to p_max.
     * @param p_max Maximum polynomial order (≤ kMaxTriOrder = 5).
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
    void SaveBinary(const std::string& path) const;
    bool LoadBinary(const std::string& path);

private:
    int max_order_ = 0;
    bool built_ = false;

    // Full prism operators (extracted from hpMusic)
    std::array<DenseMatrix, kMaxTriOrder + 1> prolong_full_;
    std::array<DenseMatrix, kMaxTriOrder + 1> restrict_full_;

    // Helpers
    void LoadProlongMatrices();
    void LoadRestrictMatrices();
};

} // namespace pmgpu

#endif // PMGPU_PRISM_RP_OPERATORS_H
