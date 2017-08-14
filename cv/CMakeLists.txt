
project(CocsoFace)

find_package(Qt5 COMPONENTS Core Gui REQUIRED)


add_executable(make_face cp.cpp)
target_link_libraries(make_face 
    Qt5::Core Qt5::Gui
)
