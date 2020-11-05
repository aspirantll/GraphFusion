set(SUBSYS_NAME fusion)
set(SUBSYS_DESC "fusion of 3D Reconstruction Steps")


cuda_add_library(${SUBSYS_NAME} STATIC
        tables.h
        cuda_scene_rep.cu
        cuda_scene_rep.h
        voxel_fusion.cpp
        voxel_fusion.h
        cuda_scene_rep.cpp)


set_target_properties(${SUBSYS_NAME}
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(${SUBSYS_NAME} datastructure tool)
