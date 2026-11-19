// HIP (AMD) instantiation of the shared GPU backend.
#include "hpmgard/DeviceShim.h"
#define HPM_MAKE_BACKEND_FN MakeHipBackend
#include "BackendDevice.inc"
