#include <map>
#include "hpmgard/Operators.h"
#include "hpmgard/Quadrature.h"
#include "ported/rp_operators.h"
#include "ported/tet_rp_operators.h"
#include "ported/prism_rp_operators.h"
#include "ported/pyramid_rp_operators.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace hpmgard {

const char* ShapeName(ElemShape s) {
    switch (s) {
    case ElemShape::QUAD: return "quadrilateral";
    case ElemShape::HEX: return "hexahedron";
    case ElemShape::TET: return "tetrahedron";
    case ElemShape::PRISM: return "wedge";
    case ElemShape::PYRAMID: return "pyramid";
    }
    return "?";
}

ElemShape ParseShape(const std::string& n) {
    std::string s; for (char c : n) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "hexahedron" || s == "hex" || s == "vtkdghex" || s == "hexa_8" || s == "hexa_27") return ElemShape::HEX;
    if (s == "tetrahedron" || s == "tet" || s == "vtkdgtet" || s == "tetra_4") return ElemShape::TET;
    if (s == "wedge" || s == "prism" || s == "vtkdgwdg" || s == "penta_6") return ElemShape::PRISM;
    if (s == "pyramid" || s == "pyr" || s == "vtkdgpyr" || s == "pyra_5") return ElemShape::PYRAMID;
    if (s == "quadrilateral" || s == "quad" || s == "vtkdgquad" || s == "quad_4") return ElemShape::QUAD;
    throw std::invalid_argument("ParseShape: unknown cell type '" + n + "'");
}

void ApplyTensor(int dim, const double* M, int n_out, int n_in, const double* in, double* out, size_t ncells) {
    const size_t nin_cell = static_cast<size_t>(std::pow(n_in, dim));
    const size_t nout_cell = static_cast<size_t>(std::pow(n_out, dim));
#pragma omp parallel
    {
        std::vector<double> t1, t2;
        if (dim == 3) { t1.resize(static_cast<size_t>(n_in) * n_in * n_out); t2.resize(static_cast<size_t>(n_in) * n_out * n_out); }
        else if (dim == 2) { t1.resize(static_cast<size_t>(n_in) * n_out); }
#pragma omp for schedule(static)
        for (size_t c = 0; c < ncells; ++c) {
            const double* x = in + c * nin_cell;
            double* y = out + c * nout_cell;
            if (dim == 3) {
                // axis i (fastest): t1[(k*n_in + j)*n_out + a]
                for (int kj = 0; kj < n_in * n_in; ++kj) {
                    const double* xs = x + static_cast<size_t>(kj) * n_in;
                    double* ts = t1.data() + static_cast<size_t>(kj) * n_out;
                    for (int a = 0; a < n_out; ++a) {
                        double s = 0.0; const double* m = M + static_cast<size_t>(a) * n_in;
                        for (int i = 0; i < n_in; ++i) s += m[i] * xs[i];
                        ts[a] = s;
                    }
                }
                // axis j: t2[(k*n_out + b)*n_out + a]
                for (int k = 0; k < n_in; ++k)
                    for (int b = 0; b < n_out; ++b) {
                        const double* m = M + static_cast<size_t>(b) * n_in;
                        double* ts = t2.data() + (static_cast<size_t>(k) * n_out + b) * n_out;
                        for (int a = 0; a < n_out; ++a) ts[a] = 0.0;
                        for (int j = 0; j < n_in; ++j) {
                            const double mj = m[j];
                            const double* t1s = t1.data() + (static_cast<size_t>(k) * n_in + j) * n_out;
                            for (int a = 0; a < n_out; ++a) ts[a] += mj * t1s[a];
                        }
                    }
                // axis k: y[(cc*n_out + b)*n_out + a]
                for (int cc = 0; cc < n_out; ++cc) {
                    const double* m = M + static_cast<size_t>(cc) * n_in;
                    double* ys = y + static_cast<size_t>(cc) * n_out * n_out;
                    for (int ba = 0; ba < n_out * n_out; ++ba) ys[ba] = 0.0;
                    for (int k = 0; k < n_in; ++k) {
                        const double mk = m[k];
                        const double* t2s = t2.data() + static_cast<size_t>(k) * n_out * n_out;
                        for (int ba = 0; ba < n_out * n_out; ++ba) ys[ba] += mk * t2s[ba];
                    }
                }
            } else if (dim == 2) {
                for (int j = 0; j < n_in; ++j)
                    for (int a = 0; a < n_out; ++a) {
                        double s = 0.0; const double* m = M + static_cast<size_t>(a) * n_in;
                        for (int i = 0; i < n_in; ++i) s += m[i] * x[static_cast<size_t>(j) * n_in + i];
                        t1[static_cast<size_t>(j) * n_out + a] = s;
                    }
                for (int b = 0; b < n_out; ++b) {
                    const double* m = M + static_cast<size_t>(b) * n_in;
                    for (int a = 0; a < n_out; ++a) {
                        double s = 0.0;
                        for (int j = 0; j < n_in; ++j) s += m[j] * t1[static_cast<size_t>(j) * n_out + a];
                        y[static_cast<size_t>(b) * n_out + a] = s;
                    }
                }
            } else {
                for (int a = 0; a < n_out; ++a) {
                    double s = 0.0; const double* m = M + static_cast<size_t>(a) * n_in;
                    for (int i = 0; i < n_in; ++i) s += m[i] * x[i];
                    y[a] = s;
                }
            }
        }
    }
}

void ApplyDense(const double* M, int n_out, int n_in, const double* in, double* out, size_t ncells) {
#pragma omp parallel for schedule(static)
    for (size_t c = 0; c < ncells; ++c) {
        const double* x = in + c * static_cast<size_t>(n_in);
        double* y = out + c * static_cast<size_t>(n_out);
        for (int a = 0; a < n_out; ++a) {
            double s = 0.0; const double* m = M + static_cast<size_t>(a) * n_in;
            for (int i = 0; i < n_in; ++i) s += m[i] * x[i];
            y[a] = s;
        }
    }
}

void OperatorSet::Restrict(const double* u, size_t ncells, double* base) const {
    if (tensor) ApplyTensor(ShapeDim(shape), RE1.data(), p_base + 1, p + 1, u, base, ncells);
    else ApplyDense(Rfull.data(), nsp_base, nsp_data, u, base, ncells);
}

void OperatorSet::Prolong(const double* base, size_t ncells, double* u) const {
    if (tensor) ApplyTensor(ShapeDim(shape), Pc1.data(), p + 1, p_base + 1, base, u, ncells);
    else ApplyDense(Pfull.data(), nsp_data, nsp_base, base, u, ncells);
}

namespace {

// Dense tables from mp-MGARD (hpMusic node sets), built once per shape.
template <class Table>
const Table& StaticTable(int p_max) {
    static Table t; static std::once_flag once;
    std::call_once(once, [&]() { t.Build(p_max); });
    if (t.MaxOrder() < p_max) throw std::runtime_error("simplex operator table built for a lower p_max than requested");
    return t;
}

template <class Table>
void ComposeDense(const Table& t, int p, int p_base, int nsp_data, int nsp_base, std::vector<double>& Rfull, std::vector<double>& Pfull) {
    // R: apply R(p) first (p -> p-1), then R(p-1), ... down to R(p_base+1).
    std::vector<double> R; int rr = 0, rc = 0;
    for (int q = p; q > p_base; --q) {
        const pmgpu::DenseMatrix& Rq = t.GetRestrictFull(q);   // (nsp(q-1) x nsp(q))
        if (R.empty()) { R = Rq.data; rr = Rq.rows; rc = Rq.cols; }
        else { R = MatMul(Rq.data, Rq.rows, Rq.cols, R, rr, rc); rr = Rq.rows; }
    }
    std::vector<double> P; int pr = 0, pc = 0;
    for (int q = p_base; q < p; ++q) {
        const pmgpu::DenseMatrix& Pq = t.GetProlongFull(q);    // (nsp(q+1) x nsp(q))
        if (P.empty()) { P = Pq.data; pr = Pq.rows; pc = Pq.cols; }
        else { P = MatMul(Pq.data, Pq.rows, Pq.cols, P, pr, pc); pr = Pq.rows; }
    }
    if (p == p_base) {
        R.assign(static_cast<size_t>(nsp_data) * nsp_data, 0.0); P = R;
        for (int i = 0; i < nsp_data; ++i) { R[static_cast<size_t>(i) * nsp_data + i] = 1.0; P[static_cast<size_t>(i) * nsp_data + i] = 1.0; }
        rr = rc = pr = pc = nsp_data;
    }
    if (rr != nsp_base || rc != nsp_data || pr != nsp_data || pc != nsp_base)
        throw std::runtime_error("simplex operator table dimensions do not match NumSP");
    Rfull = std::move(R); Pfull = std::move(P);
}

} // namespace

std::vector<double> KroneckerFull(const std::vector<double>& M1, int n_out, int n_in, int dim) {
    std::vector<double> full(1, 1.0); int ro = 1, ci = 1;
    for (int d = 0; d < dim; ++d) {
        std::vector<double> next(static_cast<size_t>(ro) * n_out * ci * n_in);
        for (int a = 0; a < n_out; ++a) for (int i = 0; i < n_in; ++i)
            for (int r = 0; r < ro; ++r) for (int c = 0; c < ci; ++c)
                next[(static_cast<size_t>(a) * ro + r) * (static_cast<size_t>(ci) * n_in) + (static_cast<size_t>(i) * ci + c)] = M1[static_cast<size_t>(a) * n_in + i] * full[static_cast<size_t>(r) * ci + c];
        full.swap(next); ro *= n_out; ci *= n_in;
    }
    return full;
}

static OperatorSet BuildOperatorsUncached(ElemShape shape, int p, int p_base, NodeSet ns, BaseRepresentation rep);

OperatorSet BuildOperators(ElemShape shape, int p, int p_base, NodeSet ns, BaseRepresentation rep) {
    static std::mutex m; static std::map<std::string, OperatorSet> cache;
    const std::string key = std::to_string(static_cast<int>(shape)) + "/" + std::to_string(p) + "/" + std::to_string(p_base) + "/" + std::to_string(static_cast<int>(ns)) + "/" + std::to_string(static_cast<int>(rep));
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    OperatorSet o = BuildOperatorsUncached(shape, p, p_base, ns, rep);
    o.key = key;
    if (o.tensor) {   // dense per-cell matrices for the fused GPU kernels
        const int dim = ShapeDim(shape);
        o.Rfull = KroneckerFull(o.RE1, p_base + 1, p + 1, dim);
        o.Pfull = KroneckerFull(o.Pc1, p + 1, p_base + 1, dim);
    }
    cache[key] = o;
    return o;
}

static OperatorSet BuildOperatorsUncached(ElemShape shape, int p, int p_base, NodeSet ns, BaseRepresentation rep) {
    if (p < 0 || p_base < 0 || p_base > p) throw std::invalid_argument("BuildOperators: need 0 <= p_base <= p");
    OperatorSet o;
    o.shape = shape; o.p = p; o.p_base = p_base; o.node_set = ns;
    o.nsp_data = NumSP(shape, p); o.nsp_base = NumSP(shape, p_base);
    o.tensor = IsTensorShape(shape);
    const int dim = ShapeDim(shape);
    if (o.tensor) {
        const int nd = p + 1, nb = p_base + 1;
        const std::vector<double> xd = Nodes(ns, nd), wd = Weights(ns, nd);
        const std::vector<double> xb = Nodes(ns, nb), wb = Weights(ns, nb);
        std::vector<double> R1 = L2RestrictMatrix(xd, wd, xb, wb);            // nb x nd
        // closed representation only changes something when the base nodes are not already closed
        const bool closed = (rep == BaseRepresentation::Closed) && (p_base < p) && (ns != NodeSet::GaussLobatto) && nb >= 2;
        o.base_rep = closed ? BaseRepresentation::Closed : BaseRepresentation::DataNodes;
        std::vector<double> store_nodes = closed ? GaussLobattoNodes(nb) : xb;
        if (p_base == p) { R1.assign(static_cast<size_t>(nd) * nd, 0.0); for (int i = 0; i < nd; ++i) R1[static_cast<size_t>(i) * nd + i] = 1.0; store_nodes = xd; }
        const std::vector<double> E1 = InterpMatrix(xb, store_nodes);         // nb x nb
        o.RE1 = (p_base == p) ? R1 : MatMul(E1, nb, nb, R1, nb, nd);          // nb x nd
        o.Pc1 = InterpMatrix(store_nodes, xd);                                // nd x nb
        o.amp_inf = std::pow(MaxRowAbsSum(o.Pc1, nd, nb), dim);
    } else {
        o.base_rep = BaseRepresentation::DataNodes;                           // hpMusic simplex node sets only
        switch (shape) {
        case ElemShape::TET:     ComposeDense(StaticTable<pmgpu::TetRPOperatorTable>(pmgpu::kMaxTetOrder), p, p_base, o.nsp_data, o.nsp_base, o.Rfull, o.Pfull); break;
        case ElemShape::PRISM:   ComposeDense(StaticTable<pmgpu::PrismRPOperatorTable>(pmgpu::kMaxTriOrder), p, p_base, o.nsp_data, o.nsp_base, o.Rfull, o.Pfull); break;
        case ElemShape::PYRAMID: ComposeDense(StaticTable<pmgpu::PyramidRPOperatorTable>(pmgpu::kMaxPyramidOrder), p, p_base, o.nsp_data, o.nsp_base, o.Rfull, o.Pfull); break;
        default: throw std::invalid_argument("BuildOperators: unsupported shape");
        }
        o.amp_inf = MaxRowAbsSum(o.Pfull, o.nsp_data, o.nsp_base);
    }
    return o;
}

} // namespace hpmgard
