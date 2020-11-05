set(SUBSYS_NAME inputsource)
set(SUBSYS_DESC "InputSource for Data")

FIND_PACKAGE(realsense2 REQUIRED)
INCLUDE_DIRECTORIES(${realsense2_INCLUDE_DIR})

add_library(${SUBSYS_NAME} STATIC
        file_inputsource.cpp
        file_inputsource.h
        realsense_inputsource.cpp
        realsense_inputsource.h
        )


target_link_libraries(${SUBSYS_NAME}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${realsense2_LIBRARY}
        datastructure
        processor
        tool
        glog
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

