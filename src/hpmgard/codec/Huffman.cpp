// Canonical Huffman coder for int32 symbol streams (CPU reference implementation).
//
// Stream: 'HUF1' u32 | n u64 | dict_bits u8 | table_count u32 | table_count x (symbol u32, length u8)
//         | outliers u64 | bitstream_bytes u64 | bitstream.
// Symbols are zigzag-mapped to unsigned values; values below 2^dict_bits are coded by the table, larger
// values by the escape symbol (index 2^dict_bits) followed by the 32-bit value. Codes are canonical
// (assigned in (length, symbol) order), limited to 32 bits, MSB first.
#include "hpmgard/codec/Codec.h"
#include "hpmgard/Bytes.h"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace hpmgard {
namespace codec {
namespace {

constexpr uint32_t kHufMagic = 0x31465548u;   // "HUF1" little-endian bytes H,U,F,1
constexpr int kMaxLen = 32;

inline uint32_t Zigzag(int32_t v) { return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31); }
inline int32_t Unzigzag(uint32_t z) { return static_cast<int32_t>((z >> 1) ^ (0u - (z & 1u))); }

// Huffman code lengths from frequencies (symbols with zero frequency get length 0).
void BuildLengths(const std::vector<uint64_t>& freq, std::vector<uint8_t>& len) {
    const size_t A = freq.size();
    len.assign(A, 0);
    std::vector<uint32_t> alive;
    for (uint32_t s = 0; s < A; ++s) if (freq[s]) alive.push_back(s);
    if (alive.empty()) return;
    if (alive.size() == 1) { len[alive[0]] = 1; return; }
    std::vector<uint64_t> f(freq);
    for (;;) {
        // nodes: 0..A-1 leaves, A.. internal
        struct Node { uint64_t w; uint32_t id; bool operator>(const Node& o) const { return w > o.w || (w == o.w && id > o.id); } };
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
        std::vector<int32_t> parent(A + alive.size(), -1);
        for (uint32_t s : alive) pq.push({f[s], s});
        uint32_t next = static_cast<uint32_t>(A);
        while (pq.size() > 1) {
            Node a = pq.top(); pq.pop(); Node b = pq.top(); pq.pop();
            parent[a.id] = static_cast<int32_t>(next); parent[b.id] = static_cast<int32_t>(next);
            pq.push({a.w + b.w, next}); ++next;
        }
        int maxlen = 0;
        for (uint32_t s : alive) { int d = 0; for (int32_t x = s; parent[x] >= 0; x = parent[x]) ++d; len[s] = static_cast<uint8_t>(d); maxlen = std::max(maxlen, d); }
        if (maxlen <= kMaxLen) return;
        for (uint32_t s : alive) f[s] = (f[s] >> 1) | 1u;   // flatten and retry
    }
}

struct BitWriter {
    std::vector<uint8_t>& out; uint64_t acc = 0; int nbits = 0;
    explicit BitWriter(std::vector<uint8_t>& o) : out(o) {}
    void put(uint32_t code, int len) {
        acc = (acc << len) | (len == 32 ? static_cast<uint64_t>(code) : (static_cast<uint64_t>(code) & ((1ull << len) - 1)));
        nbits += len;
        while (nbits >= 8) { out.push_back(static_cast<uint8_t>(acc >> (nbits - 8))); nbits -= 8; }
    }
    void flush() { if (nbits > 0) { out.push_back(static_cast<uint8_t>(acc << (8 - nbits))); nbits = 0; } }
};

struct BitReader {
    const uint8_t* p; size_t n; size_t byte = 0; int bit = 0;
    BitReader(const uint8_t* pp, size_t nn) : p(pp), n(nn) {}
    inline uint32_t get1() {
        if (byte >= n) throw std::runtime_error("HuffmanDecode: bitstream exhausted");
        const uint32_t v = (p[byte] >> (7 - bit)) & 1u;
        if (++bit == 8) { bit = 0; ++byte; }
        return v;
    }
    uint32_t get(int len) { uint32_t v = 0; for (int i = 0; i < len; ++i) v = (v << 1) | get1(); return v; }
};

} // namespace

std::vector<uint8_t> HuffmanEncode(const int32_t* sym, size_t n, int dict_bits) {
    if (dict_bits < 1 || dict_bits > 24) throw std::invalid_argument("HuffmanEncode: dict_bits must be in [1, 24]");
    const uint32_t D = 1u << dict_bits, ESC = D;
    std::vector<uint64_t> freq(D + 1, 0);
    uint64_t outliers = 0;
    for (size_t i = 0; i < n; ++i) { const uint32_t z = Zigzag(sym[i]); if (z < D) ++freq[z]; else { ++freq[ESC]; ++outliers; } }
    std::vector<uint8_t> len; BuildLengths(freq, len);
    std::vector<uint32_t> order;
    for (uint32_t s = 0; s <= D; ++s) if (len[s]) order.push_back(s);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return len[a] != len[b] ? len[a] < len[b] : a < b; });
    std::vector<uint32_t> code(D + 1, 0);
    for (size_t k = 0; k < order.size(); ++k) {
        if (k == 0) code[order[0]] = 0;
        else code[order[k]] = (code[order[k - 1]] + 1) << (len[order[k]] - len[order[k - 1]]);
    }
    ByteWriter w;
    w.u32(kHufMagic); w.u64(n); w.u8(static_cast<uint8_t>(dict_bits)); w.u32(static_cast<uint32_t>(order.size()));
    for (uint32_t s : order) { w.u32(s); w.u8(len[s]); }
    w.u64(outliers);
    const size_t size_pos = w.size(); w.u64(0);
    const size_t start = w.size();
    {
        BitWriter bw(w.buf);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t z = Zigzag(sym[i]);
            if (z < D) bw.put(code[z], len[z]);
            else { bw.put(code[ESC], len[ESC]); bw.put(z, 32); }
        }
        bw.flush();
    }
    const uint64_t bits_bytes = w.size() - start;
    for (int i = 0; i < 8; ++i) w.buf[size_pos + i] = static_cast<uint8_t>(bits_bytes >> (8 * i));
    return w.buf;
}

void HuffmanDecode(const uint8_t* b, size_t nb, int32_t* out, size_t n) {
    ByteReader r(b, nb);
    if (r.u32() != kHufMagic) throw std::runtime_error("HuffmanDecode: bad magic");
    const uint64_t nn = r.u64();
    if (nn != n) throw std::runtime_error("HuffmanDecode: count mismatch");
    const int dict_bits = r.u8(); const uint32_t D = 1u << dict_bits, ESC = D;
    const uint32_t tcount = r.u32();
    std::vector<uint32_t> order(tcount); std::vector<uint8_t> len(tcount);
    for (uint32_t k = 0; k < tcount; ++k) { order[k] = r.u32(); len[k] = r.u8(); }
    r.u64();   // outliers (informational)
    const uint64_t bits_bytes = r.u64();
    const uint8_t* bits = r.bytes(bits_bytes);
    if (n == 0) return;
    if (tcount == 0) throw std::runtime_error("HuffmanDecode: empty table for non-empty stream");
    // canonical decode tables per length
    uint32_t first_code[kMaxLen + 2] = {0}, first_index[kMaxLen + 2] = {0}, count[kMaxLen + 2] = {0};
    for (uint32_t k = 0; k < tcount; ++k) ++count[len[k]];
    {
        uint32_t code = 0, idx = 0;
        for (int L = 1; L <= kMaxLen; ++L) {
            first_code[L] = code; first_index[L] = idx;
            code = (code + count[L]) << 1; idx += count[L];
        }
    }
    BitReader br(bits, bits_bytes);
    for (size_t i = 0; i < n; ++i) {
        uint32_t code = 0; int L = 0; uint32_t s = 0;
        for (;;) {
            code = (code << 1) | br.get1(); ++L;
            if (L > kMaxLen) throw std::runtime_error("HuffmanDecode: invalid code");
            if (count[L] && code >= first_code[L] && code - first_code[L] < count[L]) { s = order[first_index[L] + (code - first_code[L])]; break; }
        }
        if (s == ESC) { const uint32_t z = br.get(32); out[i] = Unzigzag(z); }
        else out[i] = Unzigzag(s);
    }
    (void)D;
}

} // namespace codec
} // namespace hpmgard
