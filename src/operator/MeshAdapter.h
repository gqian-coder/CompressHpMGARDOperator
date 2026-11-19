// Reads the cell_grid mesh (connectivity + coordinates) of one block through ADIOS2 and turns it into
// the corner geometry the dedup base needs. One block is cached at a time (variables of a block are
// usually processed back to back).
#pragma once
#include "hpmgard/Types.h"
#include <string>

namespace hpmgard {
namespace plugin {

class MeshAdapter {
public:
    // conn_var: int64 {ncells, nodes_per_cell} (block-local ids); coords_var: double/float {npoints, 3}.
    // Returns the corners of block `blockid` of the given variables, in tensor corner order.
    const MeshCorners& Load(const std::string& meshfile, const std::string& conn_var, const std::string& coords_var,
                            size_t blockid, ElemShape shape);
    void Clear();
private:
    std::string key_;
    MeshCorners corners_;
};

} // namespace plugin
} // namespace hpmgard
