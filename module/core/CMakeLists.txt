set(SUBSYS_NAME Core)
set(SUBSYS_DESC "Core Module of 3D Reconstruction")

### ---[ Add the libraries subdirectories
include("${RTF_SOURCE_DIR}/cmake/rtf_targets.cmake")

collect_subproject_directory_names("${CMAKE_CURRENT_SOURCE_DIR}" "CMakeLists.txt" RTF_CORE_NAMES RTF_CORE_DIRS doc)
set(RTF_CORE_NAMES_UNSORTED ${RTF_CORE_NAMES})
topological_sort(RTF_CORE_NAMES RTF_ _DEPENDS)
sort_relative(RTF_CORE_NAMES_UNSORTED RTF_CORE_NAMES RTF_CORE_DIRS)
foreach(subdir ${RTF_CORE_DIRS})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
endforeach()
