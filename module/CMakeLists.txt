set(SUBSYS_NAME module)
set(SUBSYS_DESC "Modules of 3D Reconstruction")

### ---[ Add the libraries subdirectories
include("${GraphFusion_SOURCE_DIR}/cmake/rtf_targets.cmake")

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" GraphFusion_MODULE_NAMES GraphFusion_MODULE_DIRS doc)
set(GraphFusion_MODULE_NAMES_UNSORTED ${GraphFusion_MODULE_NAMES})
topological_sort(GraphFusion_MODULE_NAMES GraphFusion_ _DEPENDS)
sort_relative(GraphFusion_MODULE_NAMES_UNSORTED GraphFusion_MODULE_NAMES GraphFusion_MODULE_DIRS)
foreach(subdir ${GraphFusion_MODULE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()
