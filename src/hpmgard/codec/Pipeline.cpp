#include "hpmgard/codec/Codec.h"
#include "hpmgard/codec/MgardXHuff.h"
#include "hpmgard/Bytes.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace hpmgard {
namespace codec {

namespace {
std::string Lower(std::string s) { for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }
constexpr uint8_t kContainerMagic = 0x50;   // 'P'
}

const char* StageName(StageId id) {
    switch (id) {
    case StageId::Raw: return "raw";
    case StageId::FRBP: return "frbp";
    case StageId::Huffman: return "huffman";
    case StageId::Zstd: return "zstd";
    case StageId::MgardXHuffman: return "xhuff";
    }
    return "?";
}

std::vector<Stage> ParseChain(const std::string& spec) {
    std::vector<Stage> chain;
    size_t start = 0;
    while (start <= spec.size()) {
        size_t bar = spec.find('|', start);
        std::string tok = Lower(spec.substr(start, bar == std::string::npos ? std::string::npos : bar - start));
        tok.erase(std::remove_if(tok.begin(), tok.end(), [](unsigned char c) { return std::isspace(c); }), tok.end());
        if (!tok.empty()) {
            Stage s; std::string name = tok; int param = -1;
            const size_t colon = tok.find(':');
            if (colon != std::string::npos) { name = tok.substr(0, colon); param = std::stoi(tok.substr(colon + 1)); }
            if (name == "raw") s.id = StageId::Raw;
            else if (name == "frbp") s.id = StageId::FRBP;
            else if (name == "huffman" || name == "huff") { s.id = StageId::Huffman; s.param = param < 0 ? 16 : param; }
            else if (name == "zstd") { s.id = StageId::Zstd; s.param = param < 0 ? 3 : param; }
            else if (name == "xhuff" || name == "mgardx_huffman") { s.id = StageId::MgardXHuffman; s.param = param < 0 ? 0 : param; }
            else throw std::invalid_argument("ParseChain: unknown stage '" + name + "' in '" + spec + "'");
            chain.push_back(s);
        }
        if (bar == std::string::npos) break;
        start = bar + 1;
    }
    if (chain.empty()) throw std::invalid_argument("ParseChain: empty chain '" + spec + "'");
    if (chain[0].id == StageId::Zstd) throw std::invalid_argument("ParseChain: the first stage must code symbols (raw, frbp or huffman): '" + spec + "'");
    for (size_t i = 1; i < chain.size(); ++i)
        if (chain[i].id != StageId::Zstd) throw std::invalid_argument("ParseChain: only byte stages (zstd) may follow the first stage: '" + spec + "'");
    return chain;
}

std::string ChainString(const std::vector<Stage>& chain) {
    std::string s;
    for (size_t i = 0; i < chain.size(); ++i) {
        if (i) s += '|';
        s += StageName(chain[i].id);
        if (chain[i].id == StageId::Zstd || chain[i].id == StageId::Huffman || chain[i].id == StageId::MgardXHuffman) s += ':' + std::to_string(chain[i].param);
    }
    return s;
}

bool StageAvailable(StageId id, int device) {
    if (id == StageId::MgardXHuffman) return xhuff::Available(device);
    return true;
}

// xhuff stage payload: u8 mode (1 = MGARD-X stream follows, 0 = canonical Huffman fallback follows)
std::vector<uint8_t> EncodeSymbols(const int32_t* sym, size_t n, const std::vector<Stage>& chain, double eb_info, int device) {
    if (chain.empty()) throw std::invalid_argument("EncodeSymbols: empty chain");
    std::vector<uint8_t> cur;
    switch (chain[0].id) {
    case StageId::Raw: cur = RawEncode(sym, n); break;
    case StageId::FRBP: cur = FrbpEncode(sym, n, eb_info); break;
    case StageId::Huffman: cur = HuffmanEncode(sym, n, chain[0].param); break;
    case StageId::MgardXHuffman: {
        std::vector<uint8_t> x;
        if (xhuff::Encode(device, sym, n, chain[0].param, x)) { cur.reserve(x.size() + 1); cur.push_back(1); cur.insert(cur.end(), x.begin(), x.end()); }
        else { x = HuffmanEncode(sym, n, 16); cur.reserve(x.size() + 1); cur.push_back(0); cur.insert(cur.end(), x.begin(), x.end()); }
        break;
    }
    default: throw std::invalid_argument("EncodeSymbols: bad first stage");
    }
    std::vector<uint64_t> sizes{cur.size()};
    for (size_t i = 1; i < chain.size(); ++i) {
        cur = ZstdEncode(cur.data(), cur.size(), chain[i].param);
        sizes.push_back(cur.size());
    }
    ByteWriter w;
    w.u8(kContainerMagic);
    w.u8(static_cast<uint8_t>(chain.size()));
    for (size_t i = 0; i < chain.size(); ++i) { w.u8(static_cast<uint8_t>(chain[i].id)); w.u8(static_cast<uint8_t>(chain[i].param)); w.u64(sizes[i]); }
    w.u64(n);
    w.bytes(cur.data(), cur.size());
    return w.buf;
}

size_t PeekSymbolCount(const uint8_t* b, size_t nb) {
    ByteReader r(b, nb);
    if (r.u8() != kContainerMagic) throw std::runtime_error("DecodeSymbols: bad container magic");
    const int nstages = r.u8();
    for (int i = 0; i < nstages; ++i) { r.u8(); r.u8(); r.u64(); }
    return r.u64();
}

size_t DecodeSymbols(const uint8_t* b, size_t nb, std::vector<int32_t>& out, int device) {
    ByteReader r(b, nb);
    if (r.u8() != kContainerMagic) throw std::runtime_error("DecodeSymbols: bad container magic");
    const int nstages = r.u8();
    std::vector<Stage> chain(nstages); std::vector<uint64_t> sizes(nstages);
    for (int i = 0; i < nstages; ++i) { chain[i].id = static_cast<StageId>(r.u8()); chain[i].param = r.u8(); sizes[i] = r.u64(); }
    const uint64_t n = r.u64();
    const uint8_t* payload = r.bytes(sizes.back());
    std::vector<uint8_t> cur(payload, payload + sizes.back()), prev;
    for (int i = nstages - 1; i >= 1; --i) {
        if (chain[i].id != StageId::Zstd) throw std::runtime_error("DecodeSymbols: unexpected byte stage");
        prev.assign(sizes[i - 1], 0);
        ZstdDecode(cur.data(), cur.size(), prev.data(), prev.size());
        cur.swap(prev);
    }
    out.assign(n, 0);
    switch (chain[0].id) {
    case StageId::Raw: RawDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::FRBP: FrbpDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::Huffman: HuffmanDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::MgardXHuffman: {
        if (cur.empty()) throw std::runtime_error("DecodeSymbols: empty xhuff payload");
        if (cur[0] == 1) { if (!xhuff::Decode(device, cur.data() + 1, cur.size() - 1, out.data(), n, chain[0].param)) throw std::runtime_error("DecodeSymbols: MGARD-X Huffman decode failed (is MGARD-X available in this build?)"); }
        else HuffmanDecode(cur.data() + 1, cur.size() - 1, out.data(), n);
        break;
    }
    default: throw std::runtime_error("DecodeSymbols: bad first stage");
    }
    return n;
}

} // namespace codec
} // namespace hpmgard
