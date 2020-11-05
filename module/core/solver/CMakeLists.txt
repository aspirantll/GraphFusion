set(SUBSYS_NAME solver)
set(SUBSYS_DESC "Solver of 3D Reconstruction Steps")


add_library(${SUBSYS_NAME}
        base_solver.h
        cuda_frame.h
        cuda_math.h
        cuda_matrix.h
        matrix_conversion.h
        ransac.h)
target_link_libraries(${SUBSYS_NAME} datastructure tool)

set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
