// HIP instantiation of the MGARD-X lossless stage (compiled as HIP with MGARDX_COMPILE_HIP).
#include "hpmgard/DeviceShim.h"
#define HPM_XDEVICE mgard_x::HIP
#define HPM_XNS hip
#include "MgardXHuffDevice.inc"
