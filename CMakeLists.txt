cmake_minimum_required(VERSION 3.15)
project(SequenceMSAligner LANGUAGES C CXX)

# require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# high-optimization, native tuning, AVX2+FMA, loop unrolling, fast-math
set(PERF_FLAGS
        -O3
        -march=native
        -mtune=native
        -mavx2
        -mfma
        -funroll-loops
        -ffast-math
)

# add your executable
add_executable(msalign
        main.cpp
)

# apply PERF_FLAGS just to this target
target_compile_options(msalign PRIVATE
        -O3
        -mavx512f -mavx512bw -mavx512vl -funroll-loops
)

# find and link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(msalign PRIVATE OpenMP::OpenMP_CXX)
endif()