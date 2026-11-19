// Optional MGARD-X 1-D codec for double streams (compiled when HPMGARD_HAVE_MGARDX is defined).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hpmgard {
namespace mgardx {

bool Available();
// Compress n doubles as a 1-D array under an absolute pointwise bound (s = inf) or L2 (s = 0).
std::vector<uint8_t> Compress1D(const double* data, size_t n, double abs_tol, double s, int levels /* <=0: library default */);
void Decompress(const uint8_t* b, size_t nb, std::vector<double>& out);
// MGARD-X version compiled in (0,0,0 when unavailable).
void Version(int& major, int& minor, int& patch);

} // namespace mgardx
} // namespace hpmgard
