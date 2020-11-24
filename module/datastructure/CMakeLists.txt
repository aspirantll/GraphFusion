set(SUBSYS_NAME datastructure)
set(SUBSYS_DESC "datastructure of 3D Reconstruction: such as point, mesh, volume")



add_library(${SUBSYS_NAME} STATIC
        base_types.h
        camera.h
        camera.cpp
        frame_types.h
        frame_types.cpp
        point_types.h
        context.h
        cuda_types.h
        view_graph.cpp
        view_graph.h
        base_type.cpp
        cuda_types.cpp
        config.cpp
        config.h)




target_link_libraries(${SUBSYS_NAME}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        tool
        processor
        )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)