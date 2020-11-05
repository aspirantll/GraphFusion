# cmake
set(SUBSYS_NAME dataConverter)
set(SUBSYS_DESC "convert other data format to project data format")


#############################
# Specify the source groups #
#############################

#############################
# add executable #
#############################
add_executable(${SUBSYS_NAME} data_converter.cpp)
target_link_libraries(${SUBSYS_NAME} inputsource tool pthread)
set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
