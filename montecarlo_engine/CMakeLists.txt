cmake_minimum_required(VERSION 3.24)

project(MonteCarloEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(montecarlo_engine

    src/main.cu

    src/csv_writer.cpp
    src/wc2026_io.cpp

    src/rng/rng_engine.cu

    src/distributions/uniform_dist.cu
    src/distributions/poisson_dist.cu

    src/kernels/pi_estimation_kernel.cu
    src/kernels/poisson_sim_kernel.cu
    src/kernels/wc2026_sim_kernel.cu
)

target_compile_options(
    montecarlo_engine PRIVATE

    $<$<COMPILE_LANGUAGE:CXX>:-O3>

    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set_target_properties(
    montecarlo_engine
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
