#include "hpmgard/Quadrature.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hpmgard {

void LegendreEval(int n, double x, double& P, double& dP) {
    if (n == 0) { P = 1.0; dP = 0.0; return; }
    double p0 = 1.0, p1 = x;
    for (int k = 2; k <= n; ++k) {
        const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1; p1 = p2;
    }
    P = p1;
    const double d = x * x - 1.0;
    if (std::fabs(d) < 1e-14) {
        const double v = 0.5 * n * (n + 1.0);
        dP = (x > 0) ? v : ((n % 2) ? v : -v);
    } else {
        dP = n * (x * p1 - p0) / d;
    }
}

std::vector<double> GaussLegendreNodes(int n) {
    if (n < 1) throw std::invalid_argument("GaussLegendreNodes: n < 1");
    std::vector<double> x(n);
    if (n == 1) { x[0] = 0.0; return x; }
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i) {
        double xi = std::cos(pi * (i + 0.75) / (n + 0.5));
        for (int it = 0; it < 100; ++it) {
            double P, dP; LegendreEval(n, xi, P, dP);
            const double dx = P / dP; xi -= dx;
            if (std::fabs(dx) < 1e-16) break;
        }
        x[i] = xi;
    }
    std::sort(x.begin(), x.end());
    // symmetrise to kill rounding asymmetry
    for (int i = 0; i < n / 2; ++i) { const double m = 0.5 * (x[n - 1 - i] - x[i]); x[i] = -m; x[n - 1 - i] = m; }
    if (n % 2) x[n / 2] = 0.0;
    return x;
}

std::vector<double> GaussLegendreWeights(int n) {
    const std::vector<double> x = GaussLegendreNodes(n);
    std::vector<double> w(n);
    if (n == 1) { w[0] = 2.0; return w; }
    for (int i = 0; i < n; ++i) {
        double P, dP; LegendreEval(n, x[i], P, dP);
        w[i] = 2.0 / ((1.0 - x[i] * x[i]) * dP * dP);
    }
    return w;
}

std::vector<double> GaussLobattoNodes(int n) {
    if (n < 1) throw std::invalid_argument("GaussLobattoNodes: n < 1");
    if (n == 1) return {0.0};
    if (n == 2) return {-1.0, 1.0};
    const int m = n - 1;                      // order; interior nodes = roots of P_m'
    const double pi = 3.14159265358979323846;
    std::vector<double> x(n);
    x[0] = -1.0; x[n - 1] = 1.0;
    for (int i = 1; i < n - 1; ++i) {
        double xi = -std::cos(pi * i / m);    // Chebyshev-Lobatto initial guess
        for (int it = 0; it < 100; ++it) {
            double P, dP; LegendreEval(m, xi, P, dP);
            const double ddP = (2.0 * xi * dP - m * (m + 1.0) * P) / (1.0 - xi * xi);
            const double dx = dP / ddP; xi -= dx;
            if (std::fabs(dx) < 1e-16) break;
        }
        x[i] = xi;
    }
    std::sort(x.begin(), x.end());
    for (int i = 0; i < n / 2; ++i) { const double mid = 0.5 * (x[n - 1 - i] - x[i]); x[i] = -mid; x[n - 1 - i] = mid; }
    if (n % 2) x[n / 2] = 0.0;
    return x;
}

std::vector<double> GaussLobattoWeights(int n) {
    const std::vector<double> x = GaussLobattoNodes(n);
    std::vector<double> w(n);
    if (n == 1) { w[0] = 2.0; return w; }
    const int m = n - 1;
    for (int i = 0; i < n; ++i) {
        double P, dP; LegendreEval(m, x[i], P, dP);
        w[i] = 2.0 / (m * (m + 1.0) * P * P);
    }
    return w;
}

std::vector<double> Nodes(NodeSet ns, int n) {
    return ns == NodeSet::GaussLegendre ? GaussLegendreNodes(n) : GaussLobattoNodes(n);
}
std::vector<double> Weights(NodeSet ns, int n) {
    return ns == NodeSet::GaussLegendre ? GaussLegendreWeights(n) : GaussLobattoWeights(n);
}

double LagrangeBasis(const std::vector<double>& nodes, int j, double x) {
    double r = 1.0;
    const int n = static_cast<int>(nodes.size());
    for (int k = 0; k < n; ++k)
        if (k != j) r *= (x - nodes[k]) / (nodes[j] - nodes[k]);
    return r;
}

std::vector<double> InterpMatrix(const std::vector<double>& from, const std::vector<double>& to) {
    const int nf = static_cast<int>(from.size()), nt = static_cast<int>(to.size());
    std::vector<double> M(static_cast<size_t>(nt) * nf);
    for (int t = 0; t < nt; ++t)
        for (int f = 0; f < nf; ++f) M[static_cast<size_t>(t) * nf + f] = LagrangeBasis(from, f, to[t]);
    return M;
}

std::vector<double> L2RestrictMatrix(const std::vector<double>& hi, const std::vector<double>& w_hi,
                                     const std::vector<double>& lo, const std::vector<double>& w_lo) {
    const int nh = static_cast<int>(hi.size()), nl = static_cast<int>(lo.size());
    const std::vector<double> P = InterpMatrix(lo, hi);   // nh x nl
    std::vector<double> R(static_cast<size_t>(nl) * nh);
    for (int j = 0; j < nl; ++j)
        for (int i = 0; i < nh; ++i) R[static_cast<size_t>(j) * nh + i] = P[static_cast<size_t>(i) * nl + j] * w_hi[i] / w_lo[j];
    return R;
}

std::vector<double> MatMul(const std::vector<double>& A, int ar, int ac, const std::vector<double>& B, int br, int bc) {
    if (ac != br) throw std::invalid_argument("MatMul: dimension mismatch");
    std::vector<double> C(static_cast<size_t>(ar) * bc, 0.0);
    for (int i = 0; i < ar; ++i)
        for (int k = 0; k < ac; ++k) {
            const double a = A[static_cast<size_t>(i) * ac + k];
            for (int j = 0; j < bc; ++j) C[static_cast<size_t>(i) * bc + j] += a * B[static_cast<size_t>(k) * bc + j];
        }
    return C;
}

double MaxRowAbsSum(const std::vector<double>& M, int rows, int cols) {
    double best = 0.0;
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += std::fabs(M[static_cast<size_t>(i) * cols + j]);
        best = std::max(best, s);
    }
    return best;
}

} // namespace hpmgard
