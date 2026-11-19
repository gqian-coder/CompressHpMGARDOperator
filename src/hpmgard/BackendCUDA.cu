// CUDA (NVIDIA) instantiation of the shared GPU backend through the HIP-to-CUDA shim.
#define HPMGARD_DEVICE_CUDA 1
#include "hpmgard/DeviceShim.h"
#define HPM_MAKE_BACKEND_FN MakeCudaBackend
#include "BackendDevice.inc"
