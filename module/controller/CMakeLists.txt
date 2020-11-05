set(SUBSYS_NAME controller)
set(SUBSYS_DESC "controller of 3D Reconstruction")

cuda_add_library(${SUBSYS_NAME} STATIC
        config.h
        config.cpp
        online_reconstruction.h
        online_reconstruction.cpp
        ../tool/visual_index_hashing.cpp
        ../tool/visual_index_hashing.h
        ../tool/visual_index_hashing.cu ../tool/visual_index_hashing.cuh)
target_link_libraries(${SUBSYS_NAME} feature tool fusion registration)