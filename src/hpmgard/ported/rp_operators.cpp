/**
 * @file rp_operators.cpp
 * @brief Implementation of self-contained R/P operator tables.
 *
 * Builds restriction (L2-projection) and prolongation (Lagrange interpolation)
 * matrices from Gauss–Legendre quadrature data.  The results are bit-for-bit
 * identical to the hardcoded matrices inside hpMusic's SchemeCoef.
 */

#include "rp_operators.h"
#include <fstream>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <iomanip>

namespace pmgpu {

// ============================================================================
// DenseMatrix helpers
// ============================================================================

void DenseMatrix::Print(const std::string& label) const {
    std::cout << label << " [" << rows << " x " << cols << "]:\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(14) << std::setprecision(8) << (*this)(i, j);
        }
        std::cout << "\n";
    }
}

// ============================================================================
// Lagrange basis evaluation
// ============================================================================

double RPOperatorTable::LagrangeBasis(int j, double x,
                                       const std::vector<double>& nodes) {
    double result = 1.0;
    int n = static_cast<int>(nodes.size());
    for (int k = 0; k < n; ++k) {
        if (k != j) {
            result *= (x - nodes[k]) / (nodes[j] - nodes[k]);
        }
    }
    return result;
}

// ============================================================================
// Build 1D Prolongation:  P_{ij} = ℓ_j^{(p_lo)}(ξ_i^{(p_lo+1)})
// Maps from p_lo (p_lo+1 points) to p_lo+1 (p_lo+2 points).
// Dimensions: (p_lo+2) rows  x  (p_lo+1) cols.
// ============================================================================

DenseMatrix RPOperatorTable::BuildProlong1D(int p_lo) {
    const auto& nodes_lo = GaussNodes1D(p_lo);       // p_lo+1 points
    const auto& nodes_hi = GaussNodes1D(p_lo + 1);   // p_lo+2 points
    int n_lo = static_cast<int>(nodes_lo.size());
    int n_hi = static_cast<int>(nodes_hi.size());

    DenseMatrix P(n_hi, n_lo);
    for (int i = 0; i < n_hi; ++i) {
        for (int j = 0; j < n_lo; ++j) {
            P(i, j) = LagrangeBasis(j, nodes_hi[i], nodes_lo);
        }
    }
    return P;
}

// ============================================================================
// Build 1D Restriction (L2 projection):
//   R = M_c^{-1} * P^T * M_f
// where M_c = diag(w^{(p-1)}), M_f = diag(w^{(p)}), and P is prolong p-1 → p.
// Maps from p_hi (p_hi+1 points) to p_hi-1 (p_hi points).
// Dimensions: p_hi rows  x  (p_hi+1) cols.
// ============================================================================

DenseMatrix RPOperatorTable::BuildRestrict1D(int p_hi) {
    // P = Prolong(p_hi - 1):  maps (p_hi-1) → p_hi.
    //   P has dims: (p_hi+1) x p_hi.
    DenseMatrix P = BuildProlong1D(p_hi - 1);

    const auto& w_f = GaussWeights1D(p_hi);       // p_hi+1 weights (fine)
    const auto& w_c = GaussWeights1D(p_hi - 1);   // p_hi   weights (coarse)

    int n_f = static_cast<int>(w_f.size());   // p_hi + 1
    int n_c = static_cast<int>(w_c.size());   // p_hi

    // R = M_c^{-1} * P^T * M_f
    // R has dims: n_c x n_f  (i.e., p_hi rows x (p_hi+1) cols)
    DenseMatrix R(n_c, n_f);
    for (int i = 0; i < n_c; ++i) {
        for (int j = 0; j < n_f; ++j) {
            // (P^T * M_f)_{ij} = P_{ji} * w_f[j]
            // Then divide by w_c[i]
            R(i, j) = P(j, i) * w_f[j] / w_c[i];
        }
    }
    return R;
}

// ============================================================================
// Kronecker product  C = A ⊗ B
// ============================================================================

DenseMatrix RPOperatorTable::Kronecker(const DenseMatrix& A, const DenseMatrix& B) {
    int rA = A.rows, cA = A.cols;
    int rB = B.rows, cB = B.cols;
    DenseMatrix C(rA * rB, cA * cB);
    for (int ia = 0; ia < rA; ++ia)
        for (int ja = 0; ja < cA; ++ja)
            for (int ib = 0; ib < rB; ++ib)
                for (int jb = 0; jb < cB; ++jb)
                    C(ia * rB + ib, ja * cB + jb) = A(ia, ja) * B(ib, jb);
    return C;
}

DenseMatrix RPOperatorTable::Kronecker3(const DenseMatrix& A,
                                         const DenseMatrix& B,
                                         const DenseMatrix& C) {
    return Kronecker(A, Kronecker(B, C));
}

// ============================================================================
// Build all operators
// ============================================================================

void RPOperatorTable::Build(int dim, int p_max) {
    if (p_max > kMaxOrder) {
        throw std::runtime_error("RPOperatorTable::Build: p_max exceeds kMaxOrder");
    }
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("RPOperatorTable::Build: dim must be 1, 2, or 3");
    }

    ops_.dim = dim;
    ops_.max_order = p_max;

    // Build 1D operators for p = 0 .. p_max
    for (int p = 0; p < p_max; ++p) {
        ops_.prolong_1d[p] = BuildProlong1D(p);
    }
    for (int p = 1; p <= p_max; ++p) {
        ops_.restrict_1d[p] = BuildRestrict1D(p);
    }

    // Build full-dim operators via Kronecker products
    for (int p = 0; p < p_max; ++p) {
        const auto& P1 = ops_.prolong_1d[p];
        if (dim == 1) {
            ops_.prolong_full[p] = P1;
        } else if (dim == 2) {
            ops_.prolong_full[p] = Kronecker(P1, P1);
        } else {
            ops_.prolong_full[p] = Kronecker3(P1, P1, P1);
        }
    }
    for (int p = 1; p <= p_max; ++p) {
        const auto& R1 = ops_.restrict_1d[p];
        if (dim == 1) {
            ops_.restrict_full[p] = R1;
        } else if (dim == 2) {
            ops_.restrict_full[p] = Kronecker(R1, R1);
        } else {
            ops_.restrict_full[p] = Kronecker3(R1, R1, R1);
        }
    }

    built_ = true;
}

// ============================================================================
// CPU Apply: b[i*nv + k] += A[i][j] * x[j*nv + k]
// ============================================================================

void RPOperatorTable::ApplyRestrict(int p_hi,
                                     const double* u_hi, int n_sp_hi,
                                     double* u_lo, int& n_sp_lo,
                                     int nv) const {
    assert(built_);
    const DenseMatrix& R = ops_.restrict_full[p_hi];
    assert(R.cols == n_sp_hi);
    n_sp_lo = R.rows;

    // Zero output
    std::memset(u_lo, 0, static_cast<size_t>(n_sp_lo * nv) * sizeof(double));

    for (int i = 0; i < n_sp_lo; ++i)
        for (int j = 0; j < n_sp_hi; ++j) {
            double a = R(i, j);
            for (int k = 0; k < nv; ++k)
                u_lo[i * nv + k] += a * u_hi[j * nv + k];
        }
}

void RPOperatorTable::ApplyProlong(int p_lo,
                                    const double* u_lo, int n_sp_lo,
                                    double* u_hi, int& n_sp_hi,
                                    int nv) const {
    assert(built_);
    const DenseMatrix& P = ops_.prolong_full[p_lo];
    assert(P.cols == n_sp_lo);
    n_sp_hi = P.rows;

    std::memset(u_hi, 0, static_cast<size_t>(n_sp_hi * nv) * sizeof(double));

    for (int i = 0; i < n_sp_hi; ++i)
        for (int j = 0; j < n_sp_lo; ++j) {
            double a = P(i, j);
            for (int k = 0; k < nv; ++k)
                u_hi[i * nv + k] += a * u_lo[j * nv + k];
        }
}

// ============================================================================
// Verify R·P = I
// ============================================================================

double RPOperatorTable::VerifyRP() const {
    double max_dev = 0.0;
    for (int p = 1; p <= ops_.max_order; ++p) {
        const DenseMatrix& R = ops_.restrict_full[p];   // maps p → p-1
        const DenseMatrix& P = ops_.prolong_full[p - 1]; // maps p-1 → p
        // R·P should be identity of size NumSP(p-1) x NumSP(p-1)
        int n = R.rows;
        assert(R.cols == P.rows);  // both = NumSP(p)
        assert(P.cols == n);       // NumSP(p-1)

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double val = 0.0;
                for (int k = 0; k < R.cols; ++k) {
                    val += R(i, k) * P(k, j);
                }
                double expected = (i == j) ? 1.0 : 0.0;
                double dev = std::abs(val - expected);
                if (dev > max_dev) max_dev = dev;
            }
        }
    }
    return max_dev;
}

double RPOperatorTable::VerifyRP(int p) const {
    const DenseMatrix& R = ops_.restrict_full[p];
    const DenseMatrix& P = ops_.prolong_full[p - 1];
    int n = R.rows;
    double max_dev = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = 0.0;
            for (int k = 0; k < R.cols; ++k) {
                val += R(i, k) * P(k, j);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double dev = std::abs(val - expected);
            if (dev > max_dev) max_dev = dev;
        }
    }
    return max_dev;
}

// ============================================================================
// Print summary
// ============================================================================

void RPOperatorTable::PrintSummary() const {
    std::cout << "=== RPOperatorTable Summary ===\n"
              << "  dim      = " << ops_.dim << "D\n"
              << "  max_order= " << ops_.max_order << "\n\n";

    for (int p = 1; p <= ops_.max_order; ++p) {
        const auto& R1 = ops_.restrict_1d[p];
        const auto& Rf = ops_.restrict_full[p];
        std::cout << "  Restrict p=" << p << " → p=" << (p - 1)
                  << "  1D: " << R1.rows << "x" << R1.cols
                  << "  Full: " << Rf.rows << "x" << Rf.cols << "\n";
    }
    for (int p = 0; p < ops_.max_order; ++p) {
        const auto& P1 = ops_.prolong_1d[p];
        const auto& Pf = ops_.prolong_full[p];
        std::cout << "  Prolong  p=" << p << " → p=" << (p + 1)
                  << "  1D: " << P1.rows << "x" << P1.cols
                  << "  Full: " << Pf.rows << "x" << Pf.cols << "\n";
    }

    double rp_err = VerifyRP();
    std::cout << "  R·P = I max deviation: " << std::scientific << rp_err << "\n";
    std::cout << "==============================\n";
}

// ============================================================================
// Binary serialization
// ============================================================================

void RPOperatorTable::SaveBinary(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");

    // Header
    char magic[8] = "RPOP001";
    ofs.write(magic, 8);
    int32_t dim = ops_.dim;
    int32_t maxp = ops_.max_order;
    ofs.write(reinterpret_cast<const char*>(&dim), 4);
    ofs.write(reinterpret_cast<const char*>(&maxp), 4);

    auto writeMatrix = [&](const DenseMatrix& M) {
        int32_t r = M.rows, c = M.cols;
        ofs.write(reinterpret_cast<const char*>(&r), 4);
        ofs.write(reinterpret_cast<const char*>(&c), 4);
        ofs.write(reinterpret_cast<const char*>(M.data.data()),
                  static_cast<std::streamsize>(M.data.size() * sizeof(double)));
    };

    for (int p = 0; p < maxp; ++p)       writeMatrix(ops_.prolong_1d[p]);
    for (int p = 1; p <= maxp; ++p)       writeMatrix(ops_.restrict_1d[p]);
    for (int p = 0; p < maxp; ++p)        writeMatrix(ops_.prolong_full[p]);
    for (int p = 1; p <= maxp; ++p)       writeMatrix(ops_.restrict_full[p]);
}

bool RPOperatorTable::LoadBinary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    char magic[8];
    ifs.read(magic, 8);
    if (std::string(magic, 7) != "RPOP001") return false;

    int32_t dim, maxp;
    ifs.read(reinterpret_cast<char*>(&dim), 4);
    ifs.read(reinterpret_cast<char*>(&maxp), 4);
    ops_.dim = dim;
    ops_.max_order = maxp;

    auto readMatrix = [&](DenseMatrix& M) {
        int32_t r, c;
        ifs.read(reinterpret_cast<char*>(&r), 4);
        ifs.read(reinterpret_cast<char*>(&c), 4);
        M = DenseMatrix(r, c);
        ifs.read(reinterpret_cast<char*>(M.data.data()),
                 static_cast<std::streamsize>(M.data.size() * sizeof(double)));
    };

    for (int p = 0; p < maxp; ++p)        readMatrix(ops_.prolong_1d[p]);
    for (int p = 1; p <= maxp; ++p)        readMatrix(ops_.restrict_1d[p]);
    for (int p = 0; p < maxp; ++p)         readMatrix(ops_.prolong_full[p]);
    for (int p = 1; p <= maxp; ++p)        readMatrix(ops_.restrict_full[p]);

    built_ = true;
    return true;
}

} // namespace pmgpu
