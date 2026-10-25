cmake_minimum_required(VERSION 3.12)
project(FPV_Sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/O2 /fp:fast /W3)
else()
  add_compile_options(-O3 -ffast-math -fno-finite-math-only -Wall -Wextra)
endif()

# Core as a shared library (loaded by Python via ctypes) ...
add_library(fpvcore SHARED core/src/simulator.cpp core/src/c_api.cpp)
target_include_directories(fpvcore PUBLIC core/include)

# ... and a headless CLI for step-response CSVs / benchmarks.
add_executable(fpv_cli core/src/cli.cpp core/src/simulator.cpp)
target_include_directories(fpv_cli PRIVATE core/include)

# Drop artefacts next to each other so Python finds them on every platform.
foreach(t fpvcore fpv_cli)
  set_target_properties(${t} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
endforeach()
