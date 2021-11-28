CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

add_definitions(-DGPU_NUM_CU=16)

PROJECT(nn_gpu)
SET(CMAKE_CXX_COMPILER hipcc)

ADD_LIBRARY(nn_gpu SHARED sigmoid.cpp)

ADD_EXECUTABLE(sigmoid run_sigmoid.cpp)
TARGET_LINK_LIBRARIES(sigmoid nn_gpu)

