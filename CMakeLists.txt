cmake_minimum_required(VERSION 3.15)

project(SequenceMSAligner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(msalign src/main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(msalign PRIVATE OpenMP::OpenMP_CXX)

# Portable baseline optimization.
target_compile_options(msalign PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-funroll-loops>
)

# x86_64-only SIMD tuning. Do not apply AVX2/FMA flags on macOS arm64.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(msalign PRIVATE
            -mavx2
            -mfma
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mtune=native>
    )
endif()

# Fast-math can change numerical behavior. Keep it explicit and opt-in.
option(MSALIGN_FAST_MATH "Enable -ffast-math for maximum speed" OFF)
if(MSALIGN_FAST_MATH)
    target_compile_options(msalign PRIVATE $<$<CONFIG:Release>:-ffast-math>)
endif()

# ---- clang-format ----
find_program(CLANG_FORMAT_EXE NAMES clang-format REQUIRED)

add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${CMAKE_SOURCE_DIR}/src/main.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting src/main.cpp"
)

add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${CMAKE_SOURCE_DIR}/src/main.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Checking formatting of src/main.cpp"
)

# ---- clang-tidy ----
find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)

add_custom_target(lint
        COMMAND ${CLANG_TIDY_EXE} ${CMAKE_SOURCE_DIR}/src/main.cpp -p ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy"
)

add_custom_target(lint-fix
        COMMAND ${CLANG_TIDY_EXE} ${CMAKE_SOURCE_DIR}/src/main.cpp -p ${CMAKE_BINARY_DIR} --fix --fix-errors
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy with fixes"
)