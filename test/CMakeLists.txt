set(SUBSYS_NAME test)
set(SUBSYS_DESC "Test for 3D Reconstruction")
#set(SUBSYS_DEPS )

set(DEFAULT ON)
RTF_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")
#RTF_SUBSYS_DEPEND(build "${SUBSYS_NAME}" DEPS ${SUBSYS_DEPS} OPT_DEPS openni vtk)

#if(NOT build)
#    return()
#endif()

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" RTF_MODULE_NAMES RTF_MODULE_DIRS doc)
set(RTF_MODULE_NAMES_UNSORTED ${RTF_MODULE_NAMES})
topological_sort(RTF_MODULE_NAMES RTF_ _DEPENDS)
sort_relative(RTF_MODULE_NAMES_UNSORTED RTF_MODULE_NAMES RTF_MODULE_DIRS)
foreach(subdir ${RTF_MODULE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()