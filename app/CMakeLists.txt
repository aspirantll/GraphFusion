set(SUBSYS_NAME app)
set(SUBSYS_DESC "Applications of 3D Reconstruction")
#set(SUBSYS_DEPS )

set(DEFAULT TRUE)
RTF_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")

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
