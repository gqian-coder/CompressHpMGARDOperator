cmake_minimum_required(VERSION 3.21)
project(CompressHpMGARDOperator VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

option(HPMGARD_ENABLE_MGARDX "Link MGARD-X (mgardx stream codec and the direct-MGARD-X tuner candidate)" ON)
option(HPMGARD_BUILD_ADIOS2_PLUGIN "Build the ADIOS2 plugin operator" ON)
option(HPMGARD_ENABLE_HIP "Build the HIP (AMD GPU) backend" OFF)
option(HPMGARD_ENABLE_CUDA "CUDA backend (not yet implemented)" OFF)
if(HPMGARD_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(ZSTD REQUIRED)

# ---------------------------------------------------------------- core library
set(HPMGARD_SOURCES
  src/hpmgard/Quadrature.cpp
  src/hpmgard/Operators.cpp
  src/hpmgard/BaseSplit.cpp
  src/hpmgard/MgardX.cpp
  src/hpmgard/Header.cpp
  src/hpmgard/Compressor.cpp
  src/hpmgard/BackendCPU.cpp
  src/hpmgard/Tuner.cpp
  src/hpmgard/codec/Pipeline.cpp
  src/hpmgard/codec/RawFrbpZstd.cpp
  src/hpmgard/codec/Huffman.cpp
  src/hpmgard/codec/MgardXHuff.cpp
  src/hpmgard/ported/rp_operators.cpp
  src/hpmgard/ported/tet_rp_operators.cpp
  src/hpmgard/ported/prism_rp_operators.cpp
  src/hpmgard/ported/pyramid_rp_operators.cpp)

add_library(hpmgard STATIC ${HPMGARD_SOURCES})
add_library(hpmgard::hpmgard ALIAS hpmgard)
target_include_directories(hpmgard
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/hpmgard)
target_link_libraries(hpmgard PUBLIC OpenMP::OpenMP_CXX ZSTD::ZSTD)
target_compile_features(hpmgard PUBLIC cxx_std_17)
if(HPMGARD_ENABLE_HIP)
  target_sources(hpmgard PRIVATE src/hpmgard/BackendHIP.hip)
  set_source_files_properties(src/hpmgard/BackendHIP.hip PROPERTIES LANGUAGE HIP)
  target_compile_definitions(hpmgard PUBLIC HPMGARD_HAVE_HIP)
  target_link_libraries(hpmgard PUBLIC hip::host)
  message(STATUS "hpMGARD: HIP backend enabled (arch ${CMAKE_HIP_ARCHITECTURES})")
endif()

if(HPMGARD_ENABLE_MGARDX)
  find_package(mgard CONFIG)
  if(mgard_FOUND)
    target_link_libraries(hpmgard PUBLIC mgard::mgard)
    target_compile_definitions(hpmgard PUBLIC HPMGARD_HAVE_MGARDX)
    # The SERIAL instantiation of the MGARD-X Huffman stage lives in its own target WITHOUT OpenMP:
    # MGARD-X declares serial_dev_id with '#pragma omp threadprivate' while libmgard defines it as a
    # plain int (built without OpenMP). Compiled with OpenMP the TU would reference it as TLS and the
    # link fails (R_X86_64_TLSGD). The Cray wrapper re-appends -fopenmp, so a per-source -fno-openmp
    # does not work; a separate target does (same approach as CompressMGARDCentroidOperator).
    add_library(hpmgard_xhuff_serial STATIC src/hpmgard/codec/MgardXHuff_SERIAL.cpp)
    target_include_directories(hpmgard_xhuff_serial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(hpmgard_xhuff_serial PRIVATE mgard::mgard)
    target_compile_definitions(hpmgard_xhuff_serial PRIVATE HPMGARD_HAVE_MGARDX)
    set_target_properties(hpmgard_xhuff_serial PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(hpmgard PUBLIC hpmgard_xhuff_serial)
    if(HPMGARD_ENABLE_HIP)
      target_sources(hpmgard PRIVATE src/hpmgard/codec/MgardXHuff_HIP.hip)
      set_source_files_properties(src/hpmgard/codec/MgardXHuff_HIP.hip PROPERTIES LANGUAGE HIP COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__;MGARDX_COMPILE_HIP")
    endif()
    message(STATUS "hpMGARD: MGARD-X ${mgard_VERSION} enabled (${mgard_DIR})")
  else()
    message(WARNING "hpMGARD: MGARD-X requested but not found; the mgardx codec and the direct candidate are disabled")
  endif()
endif()

# ---------------------------------------------------------------- ADIOS2 plugin operator
if(HPMGARD_BUILD_ADIOS2_PLUGIN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/operator/CompressHpMGARDOperator.cpp")
  find_package(ADIOS2 REQUIRED)
  add_library(CompressHpMGARDOperator SHARED
    src/operator/CompressHpMGARDOperator.cpp
    src/operator/MeshAdapter.cpp)
  target_include_directories(CompressHpMGARDOperator PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_link_libraries(CompressHpMGARDOperator PUBLIC hpmgard adios2::cxx)
  install(TARGETS CompressHpMGARDOperator LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "hpMGARD: ADIOS2 plugin enabled (${ADIOS2_DIR})")
endif()

# ---------------------------------------------------------------- install + package
set(HPMGARD_INSTALL_TARGETS hpmgard)
if(TARGET hpmgard_xhuff_serial)
  list(APPEND HPMGARD_INSTALL_TARGETS hpmgard_xhuff_serial)
endif()
install(TARGETS ${HPMGARD_INSTALL_TARGETS} EXPORT hpmgardTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY src/hpmgard DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(EXPORT hpmgardTargets NAMESPACE hpmgard:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpmgard)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/hpmgardConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/hpmgardConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpmgard)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/hpmgardConfigVersion.cmake COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hpmgardConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/hpmgardConfigVersion.cmake cmake/FindZSTD.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpmgard)

message(STATUS "=== CompressHpMGARDOperator ${PROJECT_VERSION} ===")
message(STATUS "  backends: CPU (OpenMP)  HIP=${HPMGARD_ENABLE_HIP}  CUDA=${HPMGARD_ENABLE_CUDA}")
message(STATUS "  MGARD-X:  ${mgard_FOUND}")
message(STATUS "  zstd:     ${ZSTD_LIBRARY}")
