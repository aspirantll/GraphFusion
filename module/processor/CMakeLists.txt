set(SUBSYS_NAME processor)
set(SUBSYS_DESC "Processor of Data")


cuda_add_library(${SUBSYS_NAME} STATIC
        frame_converters.cpp
        frame_converters.h
        depth_filters.cpp
        depth_filters.h
        image_cuda.cu
        image_cuda.cuh
        pointcloud_generator.cu
        pointcloud_generator.h
        downsample.cpp
        downsample.h
        image_cuda.cuh
        dense_match.cu
        dense_match.cuh)


target_link_libraries(${SUBSYS_NAME}
        ${CUDA_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        datastructure
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
