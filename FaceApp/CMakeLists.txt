

project(CocsoFace)

find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

set(SRCS 
    main.cpp
    mainwindow.cpp 
    videohandler.cpp 
    helper.cpp 
    src/extractFeats.cpp
)

set(HEADERS  
    mainwindow.h 
    videohandler.h 
    helper.h 
    src/extractFeats.h
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


add_executable(FaceSpy ${SRCS} design.qrc)
target_include_directories(FaceSpy
    PRIVATE ../FaceDetection/include 
    ../FaceAlignment/include 
    ../FaceIdentification/include    
    ../FALCONN/src/include  
    ../FALCONN/external/eigen  
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(FaceSpy 
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
