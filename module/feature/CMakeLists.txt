set(SUBSYS_NAME feature)
set(SUBSYS_DESC "Feature Extractor of 3D Reconstruction")

cuda_add_library(${SUBSYS_NAME} base_feature.h feature2d.h feature3d.h feature_point.cpp feature_point.h feature_matcher.h sift_feature.cpp dense_match.cpp)
target_link_libraries(${SUBSYS_NAME} ${OpenCV_LIBRARIES} datastructure tool sift_gpu DBoW2)

set_target_properties(${SUBSYS_NAME}
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${SUBSYS_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

