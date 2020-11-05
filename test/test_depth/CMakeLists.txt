# cmake
set(SUBSYS_NAME test_depth)
add_executable(${SUBSYS_NAME} test_depth.cpp)


target_link_libraries(${SUBSYS_NAME} inputsource processor tool ${PCL_LIBRARIES} )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
