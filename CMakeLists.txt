cmake_minimum_required(VERSION 3.27)
project(fluidsim C CUDA)

set(CMAKE_C_STANDARD 11)

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include/")

link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64")

add_executable(fluidsim WIN32
        src/main.c
        src/types.h
        src/sim_math.cu
        src/sim_math.cuh
        src/render_math.cuh
        src/render_math.cu
        src/sim_parser.h
        src/sim_parser.c
)

set_target_properties(fluidsim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(fluidsim cudart_static)
