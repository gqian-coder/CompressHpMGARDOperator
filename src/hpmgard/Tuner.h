// Sample-based auto-tuning of the hierarchy (base order), base quantum, base split and codecs.
#pragma once
#include "hpmgard/Types.h"
#include <string>

namespace hpmgard {

struct TuneReport { std::string text; double best_bytes_per_value = 0.0; };

// Returns cfg with base_order / base_quantum_ratio / base_method / codecs / direct_mgardx set to the
// candidate that produced the fewest bytes on a spatially spread sample of cfg.tune_sample cells.
CompressConfig Tune(const FieldView& field, const CompressConfig& cfg, const MeshCorners* mesh, TuneReport* report);

} // namespace hpmgard
