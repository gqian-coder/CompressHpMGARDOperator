#include <cstdlib>
#include "hpmgard/BaseSplit.h"
#include "hpmgard/Bytes.h"
#include "ported/sfc.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace hpmgard {

std::vector<int> CornerPermutation(ElemShape shape) {
    switch (shape) {
    // VTK hexahedron corners: 0 (-,-,-) 1 (+,-,-) 2 (+,+,-) 3 (-,+,-) 4..7 the same at +z.
    // Tensor slot (i + 2j + 4k): slot 0 <- 0, 1 <- 1, 2 <- 3, 3 <- 2, 4 <- 4, 5 <- 5, 6 <- 7, 7 <- 6.
    case ElemShape::HEX: return {0, 1, 3, 2, 4, 5, 7, 6};
    case ElemShape::QUAD: return {0, 1, 3, 2};
    default: { std::vector<int> id(NumCorners(shape)); for (size_t i = 0; i < id.size(); ++i) id[i] = static_cast<int>(i); return id; }
    }
}

DedupGeometry BuildDedupGeometry(const MeshCorners& mesh) {
    DedupGeometry g;
    const size_t n = mesh.vertex_id.size();
    if (n == 0 || mesh.xyz.size() != 3 * n) throw std::invalid_argument("BuildDedupGeometry: inconsistent mesh corners");
    // dense group ids in increasing vertex-id order (deterministic)
    std::vector<int64_t> ids(mesh.vertex_id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    g.ngroups = static_cast<uint32_t>(ids.size());
    g.vid.resize(n);
    for (size_t i = 0; i < n; ++i)
        g.vid[i] = static_cast<uint32_t>(std::lower_bound(ids.begin(), ids.end(), mesh.vertex_id[i]) - ids.begin());
    // Hilbert order of the corner DOFs (ties broken by index inside BuildCellPerm)
    g.perm = mp_mgard::sfc::BuildCellPerm(mesh.xyz.data(), n);
    g.crc_vid = Crc32(reinterpret_cast<const uint8_t*>(g.vid.data()), g.vid.size() * 4);
    g.crc_perm = Crc32(reinterpret_cast<const uint8_t*>(g.perm.data()), g.perm.size() * 4);
    return g;
}

} // namespace hpmgard

// ---------------------------------------------------------------------------------------------------
// Dedup geometry cache
#include <chrono>
#include <list>
#include <mutex>

namespace hpmgard {

namespace {
struct GeomKey { uint64_t fp; size_t nvid, nxyz; int ncorner; bool operator==(const GeomKey& o) const { return fp == o.fp && nvid == o.nvid && nxyz == o.nxyz && ncorner == o.ncorner; } };
struct GeomEntry { GeomKey key; std::shared_ptr<const DedupGeometry> geom; size_t bytes; };
thread_local double g_last_geom_seconds = 0.0;

uint64_t ChunkedCrc(const uint8_t* p, size_t n) {
    const int nchunks = 32; uint32_t crc[nchunks] = {0};
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
        const size_t a = n * static_cast<size_t>(c) / nchunks, b = n * static_cast<size_t>(c + 1) / nchunks;
        crc[c] = Crc32(p + a, b - a);
    }
    uint64_t h = 1469598103934665603ull;
    for (int c = 0; c < nchunks; ++c) { h ^= crc[c]; h *= 1099511628211ull; }
    return h;
}
GeomKey KeyOf(const MeshCorners& m) {
    GeomKey k; k.nvid = m.vertex_id.size(); k.nxyz = m.xyz.size(); k.ncorner = m.ncorner;
    const uint64_t h1 = ChunkedCrc(reinterpret_cast<const uint8_t*>(m.vertex_id.data()), m.vertex_id.size() * sizeof(int64_t));
    const uint64_t h2 = ChunkedCrc(reinterpret_cast<const uint8_t*>(m.xyz.data()), m.xyz.size() * sizeof(double));
    k.fp = h1 ^ (h2 * 0x9E3779B97F4A7C15ull);
    return k;
}
size_t CacheCapBytes() {
    static const size_t cap = [] { const char* e = std::getenv("HPMGARD_GEOM_CACHE_MB"); const long mb = e ? std::atol(e) : 1024; return static_cast<size_t>(std::max(0l, mb)) << 20; }();
    return cap;
}
}

std::shared_ptr<const DedupGeometry> CachedDedupGeometry(const MeshCorners& mesh) {
    static std::mutex mtx; static std::list<GeomEntry> lru; static size_t total = 0;
    const GeomKey key = KeyOf(mesh);
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto it = lru.begin(); it != lru.end(); ++it)
            if (it->key == key) { lru.splice(lru.begin(), lru, it); g_last_geom_seconds = 0.0; return lru.front().geom; }
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto g = std::make_shared<const DedupGeometry>(BuildDedupGeometry(mesh));
    g_last_geom_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t bytes = (g->vid.size() + g->perm.size()) * sizeof(uint32_t);
    std::lock_guard<std::mutex> lk(mtx);
    if (bytes <= CacheCapBytes()) {
        while (total + bytes > CacheCapBytes() && !lru.empty()) { total -= lru.back().bytes; lru.pop_back(); }
        lru.push_front({key, g, bytes}); total += bytes;
    }
    return g;
}

double LastGeometryBuildSeconds() { return g_last_geom_seconds; }

} // namespace hpmgard
