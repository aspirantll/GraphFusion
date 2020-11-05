# cmake
set(SUBSYS_NAME test_online)
add_executable(${SUBSYS_NAME} test_online.cpp)


target_link_libraries(${SUBSYS_NAME} inputsource feature tool fusion registration processor ${PCL_LIBRARIES} )

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
