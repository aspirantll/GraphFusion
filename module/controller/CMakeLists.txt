set(SUBSYS_NAME controller)
set(SUBSYS_DESC "controller of 3D Reconstruction")

cuda_add_library(${SUBSYS_NAME} STATIC
        online_reconstruction.h
        online_reconstruction.cpp)
target_link_libraries(${SUBSYS_NAME} feature tool fusion registration)