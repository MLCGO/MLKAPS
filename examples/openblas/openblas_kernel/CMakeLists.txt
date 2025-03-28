cmake_minimum_required(VERSION 3.16)
project(openblas_kernel)

# Find the multi-threaded OpenBLAS version
find_library(OPENBLAS_LIB 
	NAMES 
		openblaso # Fedora
		openblas  # Ubuntu
	PATHS 
    		/usr/lib/x86_64-linux-gnu /usr/local/lib /opt/OpenBLAS/lib # Ubuntu
    		/usr/lib/ # Fedora
)

find_path(OPENBLAS_INCLUDE_DIR 
	NAMES
		cblas.h 
	PATHS 
    		/usr/include/x86_64-linux-gnu/openblas-pthread /usr/local/include /opt/OpenBLAS/include # Ubuntu
    		/usr/include/openblas # Fedora
)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check if OpenBLAS was found
if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found. Please install libopenblas-dev (Ubuntu) or openblas-devel (Fedora)")
endif()


if(NOT OPENBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "OpenBLAS headers not found. Please install libopenblas-dev (Ubuntu) or openblas-devel (Fedora)")
endif()


# Set optimization flags for Release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()


# Rules to build the openBLAS example
add_executable(openblas_kernel main.cpp)
target_include_directories(openblas_kernel PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(openblas_kernel ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)


# Rules to build the test program checking openBLAS parallelisation mode
add_executable(test_parallel test_parallel.c)
target_include_directories(test_parallel PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(test_parallel ${OPENBLAS_LIB} OpenMP::OpenMP_C)


