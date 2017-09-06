

find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)


add_executable(mkdb mkdb.cpp )
target_include_directories(mkdb
        PRIVATE
        ../../../FaceDetection/include
        ../../../FaceAlignment/include
        ../../../FaceIdentification/include
        ../../../FALCONN/src/include
        ../../../FALCONN/external/eigen
        ../
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(mkdb
        Qt5::Widgets Qt5::Core Qt5::Gui

        boost_iostreams
        boost_serialization

        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_video
        opencv_photo

        viplnet
        seeta_fa_lib
        seeta_facedet_lib

        searcher

        pthread
        )

add_executable(testdb testdb.cpp )
target_include_directories(testdb
        PRIVATE
        ../../../FaceDetection/include
        ../../../FaceAlignment/include
        ../../../FaceIdentification/include
        ../../../FALCONN/src/include
        ../../../FALCONN/external/eigen
        ../
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(testdb
        Qt5::Widgets Qt5::Core Qt5::Gui

        boost_iostreams
        boost_serialization

        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_video
        opencv_photo

        viplnet
        seeta_fa_lib
        seeta_facedet_lib

        searcher

        pthread
        )
