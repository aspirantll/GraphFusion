set(SUBSYS_NAME tool)
set(SUBSYS_DESC "Common Tools")

#The library for yaml
find_package(yaml-cpp REQUIRED)
include_directories(${YAML-CPP_INCLUDE_DIRS})
link_libraries(${YAML-CPP_LIBRARY})

cuda_add_library(${SUBSYS_NAME} STATIC
        string_util.cpp
        string_util.h
        file_util.cpp
        file_util.h
        yaml_util.cpp
        yaml_util.h
        geo_util.cpp
        geo_util.h
        point_util.cpp
        point_util.h
        image_util.cpp
        image_util.h
        math.cpp
        math.h
        random.cpp
        random.h
        timer.h
        map_reduce.cu
        map_reduce.h
        thread_pool.h
        viewer_util.cpp
        viewer_util.h
        visual_index_hashing.cpp
        visual_index_hashing.cu
        visual_index_hashing.cuh
        visual_index_hashing.h)


target_link_libraries(${SUBSYS_NAME}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        yaml-cpp
        feature
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
