// Multithreaded host memcpy (OpenMP), used by the GPU backends' pipelined transfers; compiled in a C++ unit
// because the device units are built without OpenMP.
#pragma once
#include <cstddef>
namespace hpmgard {
void ParallelMemcpy(void* dst, const void* src, size_t bytes);
}
