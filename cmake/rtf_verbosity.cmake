###############################################################################
# this file is modified from pcl_verbosity.cmake
###############################################################################

# Set GraphFusion default verbosity level from cmake

# User input default info
set(GraphFusion_VERBOSITY_LEVEL Info CACHE STRING "Set GraphFusion verbosity level. Available options are: Always Error Warn Info Debug Verbose")

if(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Info)
  set(VERBOSITY_LEVEL_INFO 1)
elseif(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Always)
  set(VERBOSITY_LEVEL_ALWAYS 1)
elseif(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Error)
  set(VERBOSITY_LEVEL_ERROR 1)
elseif(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Warn)
  set(VERBOSITY_LEVEL_WARN 1)
elseif(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Debug)
  set(VERBOSITY_LEVEL_DEBUG 1)
elseif(${GraphFusion_VERBOSITY_LEVEL} STREQUAL Verbose)
  set(VERBOSITY_LEVEL_VERBOSE 1)
else()
#  message(WARNING "Unknown verbosity level ${GraphFusion_VERBOSITY_LEVEL}. Set to Info!")
  set(VERBOSITY_LEVEL_INFO)
endif()
