cmake_minimum_required(VERSION 3.14)
project(WSKDEWrapper)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# find_package(Torch REQUIRED)

add_library(WSKDEWrapperLib STATIC
    WSKDEWrapper.cpp
)
target_include_directories(WSKDEWrapperLib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_options(WSKDEWrapperLib PRIVATE -fvisibility=hidden)
target_link_libraries(WSKDEWrapperLib PUBLIC pybind11::embed Python3::Python)

# Create cmake target installers
install(
  TARGETS WSKDEWrapperLib
  EXPORT WSKDEWrapperLibTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

# Setup target file
export(
  EXPORT WSKDEWrapperLibTargets
  FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WSKDEWrapperLibTargets.cmake"
  NAMESPACE WSKDEWrapperLib::)

# Install headers
install(FILES WSKDEWrapper.h DESTINATION include)

# Install target file
set(ConfigPackageLocation lib/cmake/WSKDEWrapperLib)
install(
  EXPORT WSKDEWrapperLibTargets
  FILE WSKDEWrapperLibTargets.cmake
  NAMESPACE WSKDEWrapperLib::
  DESTINATION ${ConfigPackageLocation})

# Install the config file
install(
  FILES cmake/WSKDEWrapperLibConfig.cmake
  DESTINATION ${ConfigPackageLocation}
  COMPONENT Devel)

# Make test executable
add_executable(WSKDEWrapperTest main.cpp)
target_link_libraries(WSKDEWrapperTest PRIVATE WSKDEWrapperLib)
target_compile_options(WSKDEWrapperTest PRIVATE -fvisibility=hidden)