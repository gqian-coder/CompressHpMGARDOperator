#include "hpmgard/HostCopy.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
namespace hpmgard {
void ParallelMemcpy(void* dst, const void* src, size_t bytes) {
    constexpr size_t kPiece = size_t(1) << 20;   // 1 MB pieces
    if (bytes <= 4 * kPiece) { std::memcpy(dst, src, bytes); return; }
    const size_t pieces = (bytes + kPiece - 1) / kPiece;
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < pieces; ++p) {
        const size_t a = p * kPiece, b = std::min(bytes, a + kPiece);
        std::memcpy(static_cast<char*>(dst) + a, static_cast<const char*>(src) + a, b - a);
    }
}
}
