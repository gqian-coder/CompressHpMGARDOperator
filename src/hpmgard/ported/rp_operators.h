/**
 * @file rp_operators.h
 * @brief Self-contained Restriction / Prolongation operator tables.
 *
 * Phase 2 of the GPU-accelerated p-multilevel pipeline.
 *
 * This module builds R and P matrices from first principles using the Gauss
 * quadrature data in gauss_quadrature.h, so it has **zero runtime dependency
 * on the hpMusic library**.  The matrices are bit-for-bit identical to the
 * hardcoded ones inside hpMusic (L2-projection restriction, Lagrange
 * interpolation prolongation).
 *
 * Construction formulae (1D, reference interval [-1,1]):
 *
 *   P_{ij}^{p→p+1} = ℓ_j^{(p)}(ξ_i^{(p+1)})       (Lagrange interpolation)
 *
 *   R^{(p)}        = (M_c)^{-1}  P^T  M_f            (L2 projection)
 *
 * where M = diag(w_i) is the Gauss weight diagonal mass matrix.
 *
 * For tensor-product elements (Quad / Hex) the full 2D/3D operators are the
 * Kronecker products of the 1D operators:
 *
 *   P_2D = P_1D ⊗ P_1D          (applied as two successive 1D sweeps)
 *
 * Storage
 * -------
 * All matrices are stored as row-major std::vector<double>.  The class provides
 * both raw-pointer access (for easy GPU upload) and a high-level
 * Apply{Restrict,Prolong}() method for CPU use.
 */

#ifndef PMGPU_RP_OPERATORS_H
#define PMGPU_RP_OPERATORS_H

#include "gauss_quadrature.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
#include <cmath>

namespace pmgpu {

/**
 * @brief Element dimensionality.
 */
enum class ElemDim { Dim1D = 1, Dim2D = 2, Dim3D = 3 };

/**
 * @brief A single dense operator matrix stored in row-major order.
 */
struct DenseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;           ///< row-major: data[i*cols + j]

    DenseMatrix() = default;
    DenseMatrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r * c), 0.0) {}

    double& operator()(int i, int j)       { return data[static_cast<size_t>(i * cols + j)]; }
    double  operator()(int i, int j) const { return data[static_cast<size_t>(i * cols + j)]; }

    const double* RawPtr() const { return data.data(); }
    double*       RawPtr()       { return data.data(); }
    size_t        Bytes()  const { return data.size() * sizeof(double); }

    /// Print matrix for debugging
    void Print(const std::string& label) const;
};

/**
 * @brief Collection of 1D R/P pairs for all p transitions, plus full-dim versions.
 */
struct RPOperatorSet {
    /// 1D prolongation: prolong_1d[p] maps p → p+1.  Dims: (p+2) × (p+1).
    std::array<DenseMatrix, kMaxOrder> prolong_1d;

    /// 1D restriction: restrict_1d[p] maps p → p-1.  Dims: p × (p+1).
    /// Index p means "restrict FROM order p TO order p-1".
    std::array<DenseMatrix, kMaxOrder + 1> restrict_1d;

    /// Full (2D or 3D) prolongation: prolong_full[p] maps p → p+1.
    std::array<DenseMatrix, kMaxOrder> prolong_full;

    /// Full (2D or 3D) restriction: restrict_full[p] maps p → p-1.
    std::array<DenseMatrix, kMaxOrder + 1> restrict_full;

    /// Dimensionality (2 for quad, 3 for hex).
    int dim = 2;

    /// Maximum order for which operators are available.
    int max_order = 0;
};

/**
 * @brief Builds and caches R/P operator tables for tensor-product elements.
 *
 * Usage:
 * @code
 *   RPOperatorTable table;
 *   table.Build(2, 5);  // 2D (quad), up to p=5
 *
 *   // CPU apply
 *   table.ApplyRestrict(p_hi, u_hi, n_sp_hi, u_lo, n_sp_lo, nv);
 *   table.ApplyProlong (p_lo, u_lo, n_sp_lo, u_hi, n_sp_hi, nv);
 *
 *   // Raw matrix for GPU upload
 *   const DenseMatrix& R = table.GetRestrict1D(p);
 *   hipMemcpy(d_R, R.RawPtr(), R.Bytes(), hipMemcpyHostToDevice);
 * @endcode
 */
class RPOperatorTable {
public:
    RPOperatorTable() = default;

    /**
     * @brief Build all 1D and full-dim operators for the given dimensionality.
     * @param dim   1, 2 (quad) or 3 (hex).
     * @param p_max Maximum polynomial order (operators built for p = 1 .. p_max).
     */
    void Build(int dim, int p_max);

    /// @name Accessors
    /// @{
    const DenseMatrix& GetProlong1D(int p)    const { return ops_.prolong_1d[p]; }
    const DenseMatrix& GetRestrict1D(int p)   const { return ops_.restrict_1d[p]; }
    const DenseMatrix& GetProlongFull(int p)  const { return ops_.prolong_full[p]; }
    const DenseMatrix& GetRestrictFull(int p) const { return ops_.restrict_full[p]; }

    int  Dim()      const { return ops_.dim; }
    int  MaxOrder() const { return ops_.max_order; }
    bool IsBuilt()  const { return built_; }
    /// @}

    /// @name CPU Apply helpers (interleaved multi-variable layout)
    /// @{

    /**
     * @brief Apply full-dimensional restriction  p → p-1.
     * @param p_hi     Source order (restricts FROM p_hi TO p_hi-1).
     * @param u_hi     Source values, interleaved: u_hi[sp * nv + v].
     * @param n_sp_hi  Number of source solution points.
     * @param u_lo     [out] Target values (resized internally).
     * @param n_sp_lo  [out] Number of target solution points.
     * @param nv       Number of interleaved variables.
     */
    void ApplyRestrict(int p_hi,
                       const double* u_hi, int n_sp_hi,
                       double* u_lo, int& n_sp_lo,
                       int nv) const;

    /**
     * @brief Apply full-dimensional prolongation  p → p+1.
     * @param p_lo     Source order (prolongates FROM p_lo TO p_lo+1).
     */
    void ApplyProlong(int p_lo,
                      const double* u_lo, int n_sp_lo,
                      double* u_hi, int& n_sp_hi,
                      int nv) const;
    /// @}

    /**
     * @brief Verify R·P = I for all levels.  Returns max deviation.
     */
    double VerifyRP() const;
    double VerifyRP(int p) const;

    /**
     * @brief Print summary of all operators (sizes, norms).
     */
    void PrintSummary() const;

    /**
     * @brief Serialize all operators to a binary file.
     * @param path  Output file path.
     */
    void SaveBinary(const std::string& path) const;

    /**
     * @brief Load operators from a binary file produced by SaveBinary().
     * @param path  Input file path.
     * @return true on success.
     */
    bool LoadBinary(const std::string& path);

private:
    RPOperatorSet ops_;
    bool built_ = false;

    // ---- helpers ----
    static DenseMatrix BuildProlong1D(int p_lo);
    static DenseMatrix BuildRestrict1D(int p_hi);
    static DenseMatrix Kronecker(const DenseMatrix& A, const DenseMatrix& B);
    static DenseMatrix Kronecker3(const DenseMatrix& A, const DenseMatrix& B, const DenseMatrix& C);

    /// Evaluate 1D Lagrange basis function j at point x, given nodes.
    static double LagrangeBasis(int j, double x, const std::vector<double>& nodes);
};

} // namespace pmgpu

#endif // PMGPU_RP_OPERATORS_H
