
project(CocsoFace)

find_package(Qt5 COMPONENTS Core Gui REQUIRED)


add_executable(make_face_2 data_2.cpp)
target_link_libraries(make_face_2
    Qt5::Core Qt5::Gui
)

add_executable(make_face_4 data_4.cpp)
target_link_libraries(make_face_4
        Qt5::Core Qt5::Gui
        )