set(SUBSYS_NAME Core)
set(SUBSYS_DESC "Core Module of 3D Reconstruction")

### ---[ Add the libraries subdirectories
include("${GraphFusion_SOURCE_DIR}/cmake/rtf_targets.cmake")

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" GraphFusion_CORE_NAMES GraphFusion_CORE_DIRS doc)
set(GraphFusion_CORE_NAMES_UNSORTED ${GraphFusion_CORE_NAMES})
topological_sort(GraphFusion_CORE_NAMES GraphFusion_ _DEPENDS)
sort_relative(GraphFusion_CORE_NAMES_UNSORTED GraphFusion_CORE_NAMES GraphFusion_CORE_DIRS)
foreach(subdir ${GraphFusion_CORE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()
