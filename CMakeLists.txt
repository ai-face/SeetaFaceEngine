
cmake_minimum_required(VERSION 2.8.4)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

include_directories(FaceDetection/include)
include_directories(FaceAlignment/include)
include_directories(FaceIdentification/include)

add_subdirectory(FaceDetection)
add_subdirectory(FaceAlignment)
add_subdirectory(FaceIdentification)