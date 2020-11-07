set(SUBSYS_NAME test)
set(SUBSYS_DESC "Test for 3D Reconstruction")
#set(SUBSYS_DEPS )

set(DEFAULT ON)
GraphFusion_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")
#GraphFusion_SUBSYS_DEPEND(build "${SUBSYS_NAME}" DEPS ${SUBSYS_DEPS} OPT_DEPS openni vtk)

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