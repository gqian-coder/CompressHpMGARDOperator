// Device API shim: the GPU sources are written against the HIP runtime API and compile unchanged for
// CUDA by mapping the names (same approach as mp-MGARD's compat header). Define HPMGARD_DEVICE_CUDA
// before including this header in a .cu translation unit; .hip units include it as is.
#pragma once
#if defined(HPMGARD_DEVICE_CUDA)
#include <cuda_runtime.h>
using hipError_t = cudaError_t;
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetDeviceCount cudaGetDeviceCount
#define hipSetDevice cudaSetDevice
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipHostMalloc(ptr, bytes, flags) cudaMallocHost(ptr, bytes)
#define hipHostMallocDefault 0
#define hipHostFree cudaFreeHost
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#define HPM_DEVICE_TYPE ::hpmgard::DeviceType::CUDA
#define HPM_DEVICE_NAME "cuda"
#else
#include <hip/hip_runtime.h>
#define HPM_DEVICE_TYPE ::hpmgard::DeviceType::HIP
#define HPM_DEVICE_NAME "hip"
#endif
