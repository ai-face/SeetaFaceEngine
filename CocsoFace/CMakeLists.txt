

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

