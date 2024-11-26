cmake_minimum_required(VERSION 3.18)

# Project information
project(OpenCLBenchmarkToKokkos LANGUAGES CXX)

# Enable C++17 (required by Kokkos)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos (ensure Kokkos is installed and in your CMake path)
find_package(Kokkos REQUIRED)

# Add source files
set(SOURCES
   src_kokkos/kernels.cpp
   src_kokkos/main.cpp
)

# Include headers
include_directories(src_kokkos)

# Define the executable
add_executable(OpenCLBenchmarkToKokkos ${SOURCES})

# Link Kokkos
target_link_libraries(OpenCLBenchmarkToKokkos Kokkos::kokkos)

# Enable Kokkos OpenMP (assuming your Kokkos was compiled with OpenMP support)
if(Kokkos_ENABLE_OPENMP)
   target_compile_definitions(OpenCLBenchmarkToKokkos PRIVATE KOKKOS_ENABLE_OPENMP)
endif()
