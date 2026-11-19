#include "hpmgard/codec/Codec.h"
#include "hpmgard/codec/MgardXHuff.h"
#include "hpmgard/MgardX.h"
#include "hpmgard/Bytes.h"
#include <algorithm>
#include <functional>
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
    case StageId::MgardXZeroRleRans: return "xrle";
    case StageId::MgardXSymbolRans: return "xrans";
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
            else if (name == "xrle" || name == "mgardx_zerorle_rans") { s.id = StageId::MgardXZeroRleRans; s.param = 0; }
            else if (name == "xrans" || name == "mgardx_symbolrans") { s.id = StageId::MgardXSymbolRans; s.param = 0; }
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
    if (IsMgardXStage(id)) return xhuff::Available(device);
    return true;
}

namespace {
xhuff::XType XTypeOf(const Stage& s) {
    if (s.id == StageId::MgardXZeroRleRans) return xhuff::XType::ZeroRleRans;
    if (s.id == StageId::MgardXSymbolRans) return xhuff::XType::SymbolRans;
    return s.param > 0 ? xhuff::XType::HuffmanZstd : xhuff::XType::Huffman;
}
}

// MGARD-X stages run on the backend device only where MGARD-X itself passes its preflight (its HIP Huffman
// is broken on Frontier login-node MI210s and faults the GPU); otherwise they run on SERIAL (device 0).
namespace {
int XDevice(int device) { return (device != 0 && (!mgardx::DeviceWorks(device) || !xhuff::Available(device))) ? 0 : device; }
}
int MgardXDevice(int device) { return XDevice(device); }
std::string DefaultChain(int device) { return XDevice(device) != 0 ? "xhuff|zstd:3" : "huffman|zstd:3"; }
std::string DefaultTuneCodecs(int device) { return XDevice(device) != 0 ? "xhuff|zstd:3,huffman|zstd:3,raw|zstd:3" : "huffman|zstd:3,raw|zstd:3,frbp|zstd:3"; }

void WarmupStage(const Stage& first, int device) {
    if (IsMgardXStage(first.id)) xhuff::Warmup(XDevice(device), XTypeOf(first));
}

namespace {
// Container: u8 'P', u8 nstages, per stage {u8 id, u8 param, u64 out_bytes}, u64 n, payload (last stage's bytes)
std::vector<uint8_t> WrapContainer(std::vector<uint8_t> cur, const std::vector<Stage>& chain, size_t n) {
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
// Inverse of the byte stages: returns the first stage's bytes, the chain and n.
std::vector<uint8_t> UnwrapContainer(const uint8_t* b, size_t nb, std::vector<Stage>& chain, uint64_t& n) {
    ByteReader r(b, nb);
    if (r.u8() != kContainerMagic) throw std::runtime_error("DecodeSymbols: bad container magic");
    const int nstages = r.u8();
    chain.assign(nstages, Stage{}); std::vector<uint64_t> sizes(nstages);
    for (int i = 0; i < nstages; ++i) { chain[i].id = static_cast<StageId>(r.u8()); chain[i].param = r.u8(); sizes[i] = r.u64(); }
    n = r.u64();
    const uint8_t* payload = r.bytes(sizes.back());
    std::vector<uint8_t> cur(payload, payload + sizes.back()), prev;
    for (int i = nstages - 1; i >= 1; --i) {
        if (chain[i].id != StageId::Zstd) throw std::runtime_error("DecodeSymbols: unexpected byte stage");
        prev.assign(sizes[i - 1], 0);
        ZstdDecode(cur.data(), cur.size(), prev.data(), prev.size());
        cur.swap(prev);
    }
    return cur;
}
std::vector<uint8_t> WithMode(uint8_t mode, const std::vector<uint8_t>& x) {
    std::vector<uint8_t> cur; cur.reserve(x.size() + 1); cur.push_back(mode); cur.insert(cur.end(), x.begin(), x.end()); return cur;
}
}

// MGARD-X stage payload: u8 mode (1 = MGARD-X stream follows, 0 = canonical Huffman fallback follows)
std::vector<uint8_t> EncodeSymbols(const int32_t* sym, size_t n, const std::vector<Stage>& chain, double eb_info, int device, const int32_t* d_sym) {
    if (chain.empty()) throw std::invalid_argument("EncodeSymbols: empty chain");
    std::vector<uint8_t> cur;
    switch (chain[0].id) {
    case StageId::Raw: cur = RawEncode(sym, n); break;
    case StageId::FRBP: cur = FrbpEncode(sym, n, eb_info); break;
    case StageId::Huffman: cur = HuffmanEncode(sym, n, chain[0].param); break;
    case StageId::MgardXHuffman:
    case StageId::MgardXZeroRleRans:
    case StageId::MgardXSymbolRans: {
        const xhuff::XType t = XTypeOf(chain[0]);
        const int xdev = XDevice(device);
        std::vector<uint8_t> x;
        bool ok = false;
        if (d_sym && xdev != 0) ok = xhuff::EncodeDevice(xdev, d_sym, n, t, chain[0].param, x);
        if (!ok) ok = xhuff::Encode(xdev, sym, n, t, chain[0].param, x);
        cur = ok ? WithMode(1, x) : WithMode(0, HuffmanEncode(sym, n, 16));
        break;
    }
    default: throw std::invalid_argument("EncodeSymbols: bad first stage");
    }
    return WrapContainer(std::move(cur), chain, n);
}

std::vector<uint8_t> EncodeSymbolsDevice(const int32_t* d_sym, size_t n, const std::vector<Stage>& chain, double eb_info, int device, const std::function<const int32_t*()>& host) {
    if (chain.empty()) throw std::invalid_argument("EncodeSymbolsDevice: empty chain");
    if (IsMgardXStage(chain[0].id)) {
        const int xdev = XDevice(device);
        if (d_sym && xdev != 0) {
            std::vector<uint8_t> x;
            if (xhuff::EncodeDevice(xdev, d_sym, n, XTypeOf(chain[0]), chain[0].param, x)) return WrapContainer(WithMode(1, x), chain, n);
        }
    }
    return EncodeSymbols(host(), n, chain, eb_info, device, nullptr);
}

size_t PeekSymbolCount(const uint8_t* b, size_t nb) {
    ByteReader r(b, nb);
    if (r.u8() != kContainerMagic) throw std::runtime_error("DecodeSymbols: bad container magic");
    const int nstages = r.u8();
    for (int i = 0; i < nstages; ++i) { r.u8(); r.u8(); r.u64(); }
    return r.u64();
}

size_t DecodeSymbols(const uint8_t* b, size_t nb, std::vector<int32_t>& out, int device) {
    std::vector<Stage> chain; uint64_t n = 0;
    std::vector<uint8_t> cur = UnwrapContainer(b, nb, chain, n);
    out.assign(n, 0);
    switch (chain[0].id) {
    case StageId::Raw: RawDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::FRBP: FrbpDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::Huffman: HuffmanDecode(cur.data(), cur.size(), out.data(), n); break;
    case StageId::MgardXHuffman:
    case StageId::MgardXZeroRleRans:
    case StageId::MgardXSymbolRans: {
        if (cur.empty()) throw std::runtime_error("DecodeSymbols: empty MGARD-X stage payload");
        if (cur[0] == 1) { if (!xhuff::Decode(XDevice(device), cur.data() + 1, cur.size() - 1, out.data(), n, XTypeOf(chain[0]), chain[0].param)) throw std::runtime_error("DecodeSymbols: MGARD-X lossless decode failed (is MGARD-X available in this build?)"); }
        else HuffmanDecode(cur.data() + 1, cur.size() - 1, out.data(), n);
        break;
    }
    default: throw std::runtime_error("DecodeSymbols: bad first stage");
    }
    return n;
}

const int32_t* DecodeSymbolsToDevice(const uint8_t* b, size_t nb, size_t& n_out, int device) {
    const int xdev = XDevice(device);
    if (xdev == 0) return nullptr;
    std::vector<Stage> chain; uint64_t n = 0;
    std::vector<uint8_t> cur = UnwrapContainer(b, nb, chain, n);
    n_out = n;
    if (!IsMgardXStage(chain[0].id) || cur.empty() || cur[0] != 1) return nullptr;
    return xhuff::DecodeDevice(xdev, cur.data() + 1, cur.size() - 1, n, XTypeOf(chain[0]), chain[0].param);
}

} // namespace codec
} // namespace hpmgard
