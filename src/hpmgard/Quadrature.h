// 1-D node sets, Lagrange bases and the 1-D restriction / prolongation / change-of-basis matrices.
#pragma once
#include "hpmgard/Types.h"
#include <vector>

namespace hpmgard {

// P_n(x) and P_n'(x).
void LegendreEval(int n, double x, double& P, double& dP);

std::vector<double> GaussLegendreNodes(int n);      // n points, ascending, in [-1, 1]
std::vector<double> GaussLegendreWeights(int n);
std::vector<double> GaussLobattoNodes(int n);       // n >= 1; n = 1 -> {0}
std::vector<double> GaussLobattoWeights(int n);
std::vector<double> Nodes(NodeSet ns, int n);
std::vector<double> Weights(NodeSet ns, int n);

double LagrangeBasis(const std::vector<double>& nodes, int j, double x);

// Row-major matrices. Interp: rows = to.size(), cols = from.size(), M[t][f] = L_f(to[t]).
std::vector<double> InterpMatrix(const std::vector<double>& from, const std::vector<double>& to);
// L2 projection from the hi node set onto the lo node set with diagonal (quadrature) mass matrices:
// R = diag(w_lo)^-1 P^T diag(w_hi), rows = lo.size(), cols = hi.size().
std::vector<double> L2RestrictMatrix(const std::vector<double>& hi, const std::vector<double>& w_hi,
                                     const std::vector<double>& lo, const std::vector<double>& w_lo);
std::vector<double> MatMul(const std::vector<double>& A, int ar, int ac, const std::vector<double>& B, int br, int bc);
double MaxRowAbsSum(const std::vector<double>& M, int rows, int cols);

} // namespace hpmgard
