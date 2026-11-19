// Little-endian fixed-width serialisation helpers and CRC32.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpmgard {

class ByteWriter {
public:
    std::vector<uint8_t> buf;
    void u8(uint8_t v) { buf.push_back(v); }
    void u16(uint16_t v) { for (int i = 0; i < 2; ++i) buf.push_back(static_cast<uint8_t>(v >> (8 * i))); }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>(v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>(v >> (8 * i))); }
    void f64(double v) { uint64_t u; std::memcpy(&u, &v, 8); u64(u); }
    void str(const std::string& s) {
        if (s.size() > 65535) throw std::invalid_argument("ByteWriter: string too long");
        u16(static_cast<uint16_t>(s.size())); buf.insert(buf.end(), s.begin(), s.end());
    }
    void bytes(const void* p, size_t n) { const uint8_t* b = static_cast<const uint8_t*>(p); buf.insert(buf.end(), b, b + n); }
    void patch_u32(size_t pos, uint32_t v) { for (int i = 0; i < 4; ++i) buf[pos + i] = static_cast<uint8_t>(v >> (8 * i)); }
    size_t size() const { return buf.size(); }
};

class ByteReader {
public:
    ByteReader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    size_t pos() const { return i_; }
    size_t remaining() const { return n_ - i_; }
    void need(size_t k) const { if (i_ + k > n_) throw std::runtime_error("ByteReader: truncated stream"); }
    uint8_t u8() { need(1); return p_[i_++]; }
    uint16_t u16() { need(2); uint16_t v = 0; for (int i = 0; i < 2; ++i) v |= static_cast<uint16_t>(p_[i_ + i]) << (8 * i); i_ += 2; return v; }
    uint32_t u32() { need(4); uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p_[i_ + i]) << (8 * i); i_ += 4; return v; }
    uint64_t u64() { need(8); uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p_[i_ + i]) << (8 * i); i_ += 8; return v; }
    double f64() { uint64_t u = u64(); double v; std::memcpy(&v, &u, 8); return v; }
    std::string str() { uint16_t n = u16(); need(n); std::string s(reinterpret_cast<const char*>(p_ + i_), n); i_ += n; return s; }
    const uint8_t* bytes(size_t n) { need(n); const uint8_t* b = p_ + i_; i_ += n; return b; }
    void skip(size_t n) { need(n); i_ += n; }
private:
    const uint8_t* p_; size_t n_; size_t i_ = 0;
};

inline uint32_t Crc32(const uint8_t* data, size_t n, uint32_t seed = 0) {
    static uint32_t table[256]; static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1); table[i] = c; }
        init = true;
    }
    uint32_t c = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

} // namespace hpmgard
