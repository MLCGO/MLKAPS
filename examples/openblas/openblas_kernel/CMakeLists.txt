cmake_minimum_required(VERSION 3.16)
project(openblas_kernel)

# Find the multi-threaded OpenBLAS version explicitly
find_library(OPENBLAS_LIB NAMES openblaso PATHS /usr/lib64 /usr/local/lib /opt/OpenBLAS/lib)
find_package(OpenMP REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

add_executable(openblas_kernel main.cpp)
target_link_libraries(openblas_kernel ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)
