cmake_minimum_required(VERSION 3.16)
project(openblas_kernel C CXX)

# Find OpenBLAS using its config file
find_package(OpenBLAS
    CONFIG
    HINTS
        /usr/lib/x86_64-linux-gnu/openblas-openmp/cmake/openblas  # Ubuntu OpenMP
        /usr/lib64/cmake/openblas                                # Fedora path
)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check what OpenBLAS provides
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_DIR}")
    # Check if the expected target exists
    if(TARGET OpenBLAS::OpenBLAS)
        message(STATUS "OpenBLAS::OpenBLAS target available")
        set(OPENBLAS_TARGET OpenBLAS::OpenBLAS)
    else()
        message(WARNING "OpenBLAS::OpenBLAS target not found, falling back to manual configuration")
        set(OPENBLAS_TARGET_FALLBACK TRUE)
    endif()
else()
    message(WARNING "OpenBLAS config not found, using manual search")
    set(OPENBLAS_TARGET_FALLBACK TRUE)
endif()

# Manual fallback if needed
if(OPENBLAS_TARGET_FALLBACK)
    find_library(OPENBLAS_LIB
        NAMES
            openblas # Ubuntu uses the same name
	    openblaso # Fedora name of the openmp version
    	PATHS
            /usr/lib/x86_64-linux-gnu/openblas-openmp/ # ubuntu openmp
            /usr/lib64/ # fedora openmp
	NO_DEFAULT_PATH  # Prevent picking system default first
    )
    find_path(OPENBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATHS
            /usr/include/x86_64-linux-gnu/openblas-openmp # ubuntu openmp
            /usr/include/openblas # fedora openmp
	NO_DEFAULT_PATH  # Prevent picking system default first
    )
    if(NOT OPENBLAS_LIB)
        message(FATAL_ERROR "OpenBLAS library not found. Install openblas-openmp (Fedora) or libopenblas-openmp-dev (Ubuntu)")
    endif()
    if(NOT OPENBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "OpenBLAS headers not found. Install openblas-devel (Fedora) or libopenblas-dev (Ubuntu)")
    endif()
    message(STATUS "Manual OpenBLAS: Library=${OPENBLAS_LIB}, Include=${OPENBLAS_INCLUDE_DIR}")
endif()

# Set optimization and OpenMP flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Rules to build the OpenBLAS example
add_executable(openblas_kernel main.cpp)
if(OPENBLAS_TARGET)
    target_link_libraries(openblas_kernel PRIVATE ${OPENBLAS_TARGET} OpenMP::OpenMP_CXX)
else()
    target_include_directories(openblas_kernel PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(openblas_kernel PRIVATE ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)
endif()

# Rules to build the test program
add_executable(test_parallel test_parallel.c)
if(OPENBLAS_TARGET)
    target_link_libraries(test_parallel PRIVATE ${OPENBLAS_TARGET} OpenMP::OpenMP_C)
else()
    target_include_directories(test_parallel PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(test_parallel PRIVATE ${OPENBLAS_LIB} OpenMP::OpenMP_C)
endif()
