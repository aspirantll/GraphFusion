# RT global CMake
cmake_minimum_required(VERSION 3.9)
#Set target C++ standard and required compiler features
#set(CMAKE_BUILD_TYPE Realese)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The target C++ standard. RTF requires C++14 or higher.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(GraphFusion VERSION 0.0.1)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# find universal dependencies
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/module/" ${CMAKE_MODULE_PATH})

# include pkgconfig
#include(FindPkgConfig)

include("${RTF_SOURCE_DIR}/cmake/rtf_verbosity.cmake")
include("${RTF_SOURCE_DIR}/cmake/rtf_options.cmake")
include("${RTF_SOURCE_DIR}/cmake/rtf_utils.cmake")
DISSECT_VERSION()
GET_OS_INFO()
SET_INSTALL_DIRS()

set(RTF_OUTPUT_LIB_DIR "${RTF_BINARY_DIR}/${LIB_INSTALL_DIR}")
set(RTF_OUTPUT_BIN_DIR "${RTF_BINARY_DIR}/${BIN_INSTALL_DIR}")
make_directory("${RTF_OUTPUT_LIB_DIR}")
make_directory("${RTF_OUTPUT_BIN_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${RTF_OUTPUT_LIB_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RTF_OUTPUT_BIN_DIR}")

###
# this is copy paste form http://www.itk.org/Wiki/CMake_RPATH_handling
# in order to always make a full statement RPATH
###
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}" is_system_dir)
if("${is_system_dir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")
endif()


### ---[ Find universal dependencies
set(CUDA_ENABLED ON)
set(CUDA_MIN_VERSION "7.0")
if(CUDA_ENABLED)
    find_package(CUDA ${CUDA_MIN_VERSION} QUIET)
endif()
if(CUDA_FOUND)
    if(CUDA_ENABLED)
        add_definitions("-DCUDA_ENABLED")

        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/module/SelectCudaComputeArch.cmake)

        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")

        # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
        # Do not show warnings if the architectures are deprecated.
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
        # Set C++ 11
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11)

        message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                " archs: ${CUDA_ARCH_FLAGS_readable})")

        include_directories(${CUDA_INCLUDE_DIRS})
    else()
        set(CUDA_FOUND OFF)
        message(STATUS "Disabling CUDA support")
    endif()
else()
    set(CUDA_ENABLED OFF)
    if(CUDA_VERSION_STRING)
        message(STATUS "Disabling CUDA support (found version "
                "${CUDA_VERSION_STRING} but >= ${CUDA_MIN_VERSION} required)")
    else()
        message(STATUS "Disabling CUDA support")
    endif()
endif()

find_package(OpenMP COMPONENTS C CXX)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if(${CMAKE_VERSION} VERSION_LESS "3.7")
        message(STATUS "Found OpenMP")
    else()
        # We could use OpenMP_CXX_VERSION starting from CMake 3.9, but this value is only available on first run of CMake (see https://gitlab.kitware.com/cmake/cmake/issues/19150),
        # so we use always OpenMP_CXX_SPEC_DATE, which is available since CMake 3.7.
        message(STATUS "Found OpenMP, spec date ${OpenMP_CXX_SPEC_DATE}")
    endif()
    if(MSVC)
        if(MSVC_VERSION EQUAL 1900)
            set(OPENMP_DLL VCOMP140)
        elseif(MSVC_VERSION MATCHES "^191[0-9]$")
            set(OPENMP_DLL VCOMP140)
        elseif(MSVC_VERSION MATCHES "^192[0-9]$")
            set(OPENMP_DLL VCOMP140)
        endif()
        if(OPENMP_DLL)
            set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
        else()
            message(WARNING "Delay loading flag for OpenMP DLL is invalid.")
        endif()
    endif()
else()
    message(STATUS "Not found OpenMP")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBRARY})

#The library for logging
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
link_libraries(${GLOG_LIBRARIES})

#The library for lie
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS})
link_libraries(${Sophus_LIBRARIES})

#The library for open gl
if (POLICY CMP0072)
    set(OpenGL_GL_PREFERENCE LEGACY)
endif()
find_package(OpenGL REQUIRED)
include_directories( ${OpenGL_INCLUDE_DIRS})
link_libraries(${OpenGL_LIBRARIES})

#The library for glew
find_package(Glew REQUIRED)
include_directories( ${Glew_INCLUDE_DIRS})
link_libraries(${Glew_LIBRARIES})

include_directories(thirdparty/DBoW2)
include_directories(thirdparty)


# PCL
find_package(PCL REQUIRED)
if(PCL_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PCL_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PCL_CXX_FLAGS}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARIES})
    list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
    add_definitions(${PCL_DEFINITIONS})
endif()



### ---[ Add the libraries subdirectories
include("${RTF_SOURCE_DIR}/cmake/rtf_targets.cmake")

collect_subproject_directory_names("${RTF_SOURCE_DIR}" "CMakeLists.txt" RTF_MODULES_NAMES RTF_MODULES_DIRS doc)
set(RTF_MODULES_NAMES_UNSORTED ${RTF_MODULES_NAMES})
topological_sort(RTF_MODULES_NAMES RTF_ _DEPENDS)
sort_relative(RTF_MODULES_NAMES_UNSORTED RTF_MODULES_NAMES RTF_MODULES_DIRS)
foreach(subdir ${RTF_MODULES_DIRS})
    add_subdirectory("${RTF_SOURCE_DIR}/${subdir}")
endforeach()