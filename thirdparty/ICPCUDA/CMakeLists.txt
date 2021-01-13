cmake_minimum_required(VERSION 2.6.0)

file(GLOB srcs *.cpp *.cu *.h *.cuh)
file(GLOB cuda Cuda/*.cu Cuda/*.cpp Cuda/*.cuh Cuda/*.h)
file(GLOB containers Cuda/containers/*.cpp Cuda/containers/*.h Cuda/containers/*.cu Cuda/containers/*.cuh)

cuda_add_library(ICP STATIC
        ${srcs}
        ${cuda}
        ${cuda_objs}
        ${containers})

target_link_libraries(ICP
        ${CUDA_LIBRARIES}
)

