include("${CMAKE_CURRENT_LIST_DIR}/WSKDEWrapperLibTargets.cmake")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

set(WSKDEWRAPPER_LIBRARIES WSKDEWrapperLib::WSKDEWrapperLib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WSKDEWrapperLib DEFAULT_MSG WSKDEWRAPPER_LIBRARIES)

