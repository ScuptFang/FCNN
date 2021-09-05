cmake_minimum_required(VERSION 3.12)
project(fncc)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(fncc 
src/FC_BPNN.cpp
src/test.cpp
)
