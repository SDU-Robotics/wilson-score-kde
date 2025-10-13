cmake_minimum_required(VERSION 3.28)
project(WSKDEWrapperLibExample)

find_package(WSKDEWrapperLib REQUIRED)

add_executable(WSKDEWrapperLibExample main.cpp)
target_link_libraries(WSKDEWrapperLibExample PRIVATE WSKDEWrapperLib::WSKDEWrapperLib)
