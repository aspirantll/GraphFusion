set(SUBSYS_NAME app)
set(SUBSYS_DESC "Applications of 3D Reconstruction")
#set(SUBSYS_DEPS )

set(DEFAULT TRUE)
GraphFusion_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")

#if(NOT build)
#    return()
#endif()

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" GraphFusion_MODULE_NAMES GraphFusion_MODULE_DIRS doc)
set(GraphFusion_MODULE_NAMES_UNSORTED ${GraphFusion_MODULE_NAMES})
topological_sort(GraphFusion_MODULE_NAMES GraphFusion_ _DEPENDS)
sort_relative(GraphFusion_MODULE_NAMES_UNSORTED GraphFusion_MODULE_NAMES GraphFusion_MODULE_DIRS)
foreach(subdir ${GraphFusion_MODULE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()
