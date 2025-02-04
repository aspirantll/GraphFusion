set(SUBSYS_NAME registration)
set(SUBSYS_DESC "estimate pose of camera for every frame")

cuda_add_library(${SUBSYS_NAME} STATIC
        bundle_adjustment.cpp
        bundle_adjustment.cu
        bundle_adjustment.cuh
        global_registration.cpp
        local_registration.cpp
        registrations.h
        pnp_registration.cpp
        optimizer.cpp
        optimizer.h
        icp_cuda.cpp)
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})
link_libraries(${Ceres_LIBRARIES})

target_link_libraries(${SUBSYS_NAME}
        ${CUDA_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        datastructure
        tool
        ICP
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
