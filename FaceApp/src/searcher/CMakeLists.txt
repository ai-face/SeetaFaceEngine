

find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)


set(src_files
        FaceEngine.cpp
        FaceEngine.h
        FeaFile.cpp
        FeaFile.h
        SearchActor.cpp
        SearchActorCreate.cpp
        SearchActor.h
        )

add_library(searcher SHARED ${src_files})

target_include_directories(searcher
        PRIVATE
        ../../../FaceDetection/include
        ../../../FaceAlignment/include
        ../../../FaceIdentification/include
        ../../../FALCONN/src/include
        ../../../FALCONN/external/eigen
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(searcher
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

        pthread
        )