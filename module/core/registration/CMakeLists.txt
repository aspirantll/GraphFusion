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
        optimizer.cpp
        optimizer.h)

#The library for logging
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
link_libraries(${G2O_LIBRARIES})

SET(G2O_LIBS g2o_core g2o_types_slam3d g2o_stuff g2o_solver_eigen)

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})
link_libraries(${Ceres_LIBRARIES})

target_link_libraries(${SUBSYS_NAME}
        ${CUDA_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        ${G2O_LIBS}
        datastructure
        tool
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
