set(SUBSYS_NAME module)
set(SUBSYS_DESC "Modules of 3D Reconstruction")

### ---[ Add the libraries subdirectories
include("${RTF_SOURCE_DIR}/cmake/rtf_targets.cmake")

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" RTF_MODULE_NAMES RTF_MODULE_DIRS doc)
set(RTF_MODULE_NAMES_UNSORTED ${RTF_MODULE_NAMES})
topological_sort(RTF_MODULE_NAMES RTF_ _DEPENDS)
sort_relative(RTF_MODULE_NAMES_UNSORTED RTF_MODULE_NAMES RTF_MODULE_DIRS)
foreach(subdir ${RTF_MODULE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()
