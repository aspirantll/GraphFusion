set(SUBSYS_NAME registration)
set(SUBSYS_DESC "estimate pose of camera for every frame")

cuda_add_library(${SUBSYS_NAME} STATIC
        bundle_adjustment.cpp
        bundle_adjustment.cu
        bundle_adjustment.cuh
        CMakeLists.txt
        epipolar_geometry.cpp
        epipolar_geometry.cu
        epipolar_geometry.cuh
        global_registration.cpp
        local_registration.cpp
        registrations.h
        pnp_registration.cpp
        homography.cpp
        icp_multiview.cpp icp_multiview.cu icp_multiview.cuh)


target_link_libraries(${SUBSYS_NAME}
        ${CUDA_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_cusolver_LIBRARY}
        datastructure
        tool
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
