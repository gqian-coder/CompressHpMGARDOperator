#include "hpmgard/Header.h"
#include "hpmgard/Bytes.h"
#include <stdexcept>
#include <string>

namespace hpmgard {

namespace {
constexpr char kMagic[4] = {'H', 'P', 'M', 'G'};
constexpr size_t kPreambleBytes = 4 + 1 + 3 + 3 + 1 + 4 + 4 + 2;   // 22
}

std::vector<uint8_t> SerializeHeader(const BlockHeader& h) {
    ByteWriter m;   // metadata body
    m.u64(h.blockid); m.u64(h.dims.size()); for (uint64_t d : h.dims) m.u64(d); m.u8(h.data_type);
    m.u8(static_cast<uint8_t>(h.shape)); m.u8(h.p); m.u16(h.nsp); m.u8(h.ncomp); m.u8(h.dof_layout); m.u8(static_cast<uint8_t>(h.node_set));
    m.u8(h.base_order); m.u8(static_cast<uint8_t>(h.base_rep)); m.u8(static_cast<uint8_t>(h.base_method)); m.u8(static_cast<uint8_t>(h.norm));
    m.f64(h.tolerance);
    if (h.var_scale.size() != h.ncomp || h.tol_abs.size() != h.ncomp || h.base_quantum.size() != h.ncomp || h.residual_quantum.size() != h.ncomp)
        throw std::invalid_argument("SerializeHeader: per-component vectors must have ncomp entries");
    for (int c = 0; c < h.ncomp; ++c) { m.f64(h.var_scale[c]); m.f64(h.tol_abs[c]); m.f64(h.base_quantum[c]); m.f64(h.residual_quantum[c]); }
    m.f64(h.amp_inf);
    m.str(h.base_codec); m.str(h.residual_codec); m.u8(h.direct_mgardx);
    uint16_t flags = h.flags & 1u;
    if (!h.conn_var.empty() || !h.coords_var.empty() || h.num_vertex_groups > 0) flags |= 2u;   // mesh identity / dedup geometry
    if (h.crc_vid || h.crc_perm) flags |= 4u;
    if (!h.streams.empty()) flags |= 8u;
    if (flags & 2u) { m.str(h.conn_var); m.str(h.coords_var); m.u8(h.corner_perm_code); m.u64(h.num_vertex_groups); }
    if (flags & 4u) { m.u32(h.crc_vid); m.u32(h.crc_perm); }
    m.str(h.writer); m.u8(h.mgardx_version[0]); m.u8(h.mgardx_version[1]); m.u8(h.mgardx_version[2]); m.u8(h.writer_backend);
    if (flags & 8u) {
        m.u32(static_cast<uint32_t>(h.streams.size()));
        for (const StreamRec& s : h.streams) { m.u8(s.component); m.u8(static_cast<uint8_t>(s.kind)); m.u8(static_cast<uint8_t>(s.codec)); m.u64(s.offset); m.u64(s.bytes); m.u64(s.count); }
    }
    ByteWriter w;
    w.bytes(kMagic, 4); w.u8(kOperatorTypeByte);
    for (int i = 0; i < 3; ++i) w.u8(h.software_version[i]);
    for (int i = 0; i < 3; ++i) w.u8(h.file_version[i]);
    w.u8(1);   // little-endian
    w.u32(static_cast<uint32_t>(m.size()));
    w.u32(Crc32(m.buf.data(), m.buf.size()));
    w.u16(flags);
    w.bytes(m.buf.data(), m.buf.size());
    return w.buf;
}

bool PeekVersions(const uint8_t* b, size_t n, uint8_t software[3], uint8_t file[3]) {
    if (n < kPreambleBytes) return false;
    if (b[0] != 'H' || b[1] != 'P' || b[2] != 'M' || b[3] != 'G' || b[4] != kOperatorTypeByte) return false;
    for (int i = 0; i < 3; ++i) { software[i] = b[5 + i]; file[i] = b[8 + i]; }
    return true;
}

size_t ParseHeader(const uint8_t* b, size_t n, BlockHeader& h) {
    ByteReader r(b, n);
    const uint8_t* mg = r.bytes(4);
    if (mg[0] != 'H' || mg[1] != 'P' || mg[2] != 'M' || mg[3] != 'G') throw std::runtime_error("ParseHeader: not an hpMGARD block (bad magic)");
    const uint8_t type = r.u8();
    if (type != kOperatorTypeByte) throw std::runtime_error("ParseHeader: operator type byte " + std::to_string(type) + " != 54");
    for (int i = 0; i < 3; ++i) h.software_version[i] = r.u8();
    for (int i = 0; i < 3; ++i) h.file_version[i] = r.u8();
    if (h.file_version[0] > kFileVersion[0])
        throw std::runtime_error("ParseHeader: file version " + std::to_string(h.file_version[0]) + "." + std::to_string(h.file_version[1]) + "." + std::to_string(h.file_version[2]) + " is newer than this reader (" + std::to_string(kFileVersion[0]) + "." + std::to_string(kFileVersion[1]) + "." + std::to_string(kFileVersion[2]) + ")");
    if (r.u8() != 1) throw std::runtime_error("ParseHeader: big-endian streams are not supported");
    const uint32_t msize = r.u32(); const uint32_t crc = r.u32(); h.flags = r.u16();
    const uint8_t* mbytes = r.bytes(msize);
    if (Crc32(mbytes, msize) != crc) throw std::runtime_error("ParseHeader: metadata CRC mismatch (corrupted block)");
    ByteReader m(mbytes, msize);
    h.blockid = m.u64(); const uint64_t nd = m.u64(); h.dims.resize(nd); for (uint64_t i = 0; i < nd; ++i) h.dims[i] = m.u64(); h.data_type = m.u8();
    h.shape = static_cast<ElemShape>(m.u8()); h.p = m.u8(); h.nsp = m.u16(); h.ncomp = m.u8(); h.dof_layout = m.u8(); h.node_set = static_cast<NodeSet>(m.u8());
    h.base_order = m.u8(); h.base_rep = static_cast<BaseRepresentation>(m.u8()); h.base_method = static_cast<BaseMethod>(m.u8()); h.norm = static_cast<Norm>(m.u8());
    h.tolerance = m.f64();
    h.var_scale.resize(h.ncomp); h.tol_abs.resize(h.ncomp); h.base_quantum.resize(h.ncomp); h.residual_quantum.resize(h.ncomp);
    for (int c = 0; c < h.ncomp; ++c) { h.var_scale[c] = m.f64(); h.tol_abs[c] = m.f64(); h.base_quantum[c] = m.f64(); h.residual_quantum[c] = m.f64(); }
    h.amp_inf = m.f64();
    h.base_codec = m.str(); h.residual_codec = m.str(); h.direct_mgardx = m.u8();
    if (h.flags & 2u) { h.conn_var = m.str(); h.coords_var = m.str(); h.corner_perm_code = m.u8(); h.num_vertex_groups = m.u64(); }
    if (h.flags & 4u) { h.crc_vid = m.u32(); h.crc_perm = m.u32(); }
    h.writer = m.str(); for (int i = 0; i < 3; ++i) h.mgardx_version[i] = m.u8(); h.writer_backend = m.u8();
    h.streams.clear();
    if (h.flags & 8u) {
        const uint32_t ns = m.u32(); h.streams.resize(ns);
        for (uint32_t i = 0; i < ns; ++i) { StreamRec& s = h.streams[i]; s.component = m.u8(); s.kind = static_cast<StreamKind>(m.u8()); s.codec = static_cast<StreamCodec>(m.u8()); s.offset = m.u64(); s.bytes = m.u64(); s.count = m.u64(); }
    }
    return r.pos();
}

} // namespace hpmgard
