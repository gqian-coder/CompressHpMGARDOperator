// ADIOS2 plugin operator around the hpMGARD compressor.
//
// Parameters (ADIOS2 operator Params; the runtime knobs can be overridden by HPMGARD_<NAME> env vars):
//   cell_type (hexahedron|tetrahedron|wedge|pyramid|vtkDGHex...)  order (p)  [components]  [node_set gauss|lobatto]
//   tolerance  [mode ABS|REL_L2|REL_RANGE]  [tuning auto|fixed]  [tuning_sample]  [tune_codecs]
//   [base_order] [base_method cellavg|dedup_sfc|plain] [base_quantum] [base_codec] [residual_codec]
//   [meshfile] [connectivity_variable] [coordinates_variable]  (dedup; on decode optional, see README)
//   blockid  [threshold] [threads] [verbose]
#pragma once
#include "adios2.h"
#include "adios2/plugin/PluginOperatorInterface.h"
#include "operator/MeshAdapter.h"
#include <mutex>
#include <string>

namespace adios2 {
namespace plugin {

class CompressHpMGARDOperator : public PluginOperatorInterface {
public:
    CompressHpMGARDOperator(const Params& parameters);
    ~CompressHpMGARDOperator() override = default;

    void AddExtraParameters(const Params& params) override;
    size_t Operate(const char* dataIn, const Dims& blockStart, const Dims& blockCount, const DataType type, char* bufferOut) override;
    size_t InverseOperate(const char* bufferIn, const size_t sizeIn, char* dataOut) override;
    bool IsDataTypeValid(const DataType type) const override;
    size_t GetEstimatedSize(const size_t ElemCount, const size_t ElemSize, const size_t ndims, const size_t* dims) const override;

private:
    std::string Param(const std::string& name, const std::string& def = "") const;
    std::string m_EngineName, m_VariableName;
    hpmgard::plugin::MeshAdapter m_Mesh;
};

} // namespace plugin
} // namespace adios2

extern "C" {
adios2::plugin::CompressHpMGARDOperator* OperatorCreate(const adios2::Params& parameters);
void OperatorDestroy(adios2::plugin::CompressHpMGARDOperator* obj);
}
