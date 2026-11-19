// CUDA instantiation of the MGARD-X lossless stage (compiled with MGARDX_COMPILE_CUDA).
#define HPMGARD_DEVICE_CUDA 1
#include "hpmgard/DeviceShim.h"
#define HPM_XDEVICE mgard_x::CUDA
#define HPM_XNS cuda
#include "MgardXHuffDevice.inc"
