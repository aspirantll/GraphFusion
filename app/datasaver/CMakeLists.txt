# cmake
set(SUBSYS_NAME datasaver)
set(SUBSYS_DESC "pick the rgbd data from cameras")


#############################
# Specify the source groups #
#############################

#############################
# add executable #
#############################
add_executable(${SUBSYS_NAME} data_saver.cpp)
target_link_libraries(${SUBSYS_NAME} inputsource tool pthread)
set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
