// Restriction (data -> base), change of basis to the closed node set, and prolongation (base -> data)
// for one (shape, p, p_base, node set) combination. Tensor shapes use separable 1-D sweeps, the
// simplex shapes use the dense hpMusic-derived tables copied from mp-MGARD.
#pragma once
#include "hpmgard/Types.h"
#include <vector>

namespace hpmgard {

struct OperatorSet {
    ElemShape shape = ElemShape::HEX;
    int p = 0, p_base = 0;
    NodeSet node_set = NodeSet::GaussLegendre;
    BaseRepresentation base_rep = BaseRepresentation::Closed;   // as actually used
    int nsp_data = 0, nsp_base = 0;
    bool tensor = true;
    // tensor shapes: 1-D matrices applied along every axis
    std::vector<double> RE1;   // (n_base x n_data): L2 restriction followed by the change to closed nodes
    std::vector<double> Pc1;   // (n_data x n_base): interpolation from the base nodes back to the data nodes
    // simplex shapes: dense per-cell matrices
    std::vector<double> Rfull; // (nsp_base x nsp_data)
    std::vector<double> Pfull; // (nsp_data x nsp_base)
    double amp_inf = 1.0;      // infinity norm of the full prolongation (error amplification of base noise)

    // u: [ncells][nsp_data] (one component, contiguous) -> base: [ncells][nsp_base]
    void Restrict(const double* u, size_t ncells, double* base) const;
    // base: [ncells][nsp_base] -> u: [ncells][nsp_data]
    void Prolong(const double* base, size_t ncells, double* u) const;
};

OperatorSet BuildOperators(ElemShape shape, int p, int p_base, NodeSet ns, BaseRepresentation rep);

// Separable application of the same 1-D matrix M (n_out x n_in, row-major) along every axis of a
// dim-dimensional tensor per cell; index = i + n*j + n*n*k (first axis fastest).
void ApplyTensor(int dim, const double* M, int n_out, int n_in, const double* in, double* out, size_t ncells);
// out[c] = M (n_out x n_in) * in[c]
void ApplyDense(const double* M, int n_out, int n_in, const double* in, double* out, size_t ncells);

} // namespace hpmgard
