
cmake_minimum_required(VERSION 2.8.4)

project(FaceSpyPrj)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

if(_MSC_VER)
add_compile_options( /bigobj )
set(OpenCV_DIR E:/opt/opencv/opencv/build/ )
endif()

find_package(OpenCV)

if(_MSC_VER)
set(Qt5_DIR C:/Qt/Qt5.9.0/5.9/msvc2015_64/lib/cmake/Qt5)
endif()

if(_MSC_VER)
set(BOOST_ROOT E:/opt/boost.install.dll) # install prefix for boost
add_definitions( -D_WIN32_WINNT=0x0601 )
add_definitions( -DBOOST_ALL_DYN_LINK  ) # for shared boost libs
endif()

#add_definitions( -D__VIPL_LOG__ )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


find_package(Boost)
link_directories(${Boost_LIBRARY_DIRS})

include_directories(FaceDetection/include)
include_directories(FaceAlignment/include)
include_directories(FaceIdentification/include)

add_subdirectory(FaceDetection)
add_subdirectory(FaceAlignment)
add_subdirectory(FaceIdentification)
add_subdirectory(FaceApp)
add_subdirectory(cv)
