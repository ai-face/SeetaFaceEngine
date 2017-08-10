aux_source_directory (. SRC_LIST_TEST)

message(${SRC_LIST_TEST})

# add external libraries
find_package(OpenCV REQUIRED)


foreach( f ${tests})
   add_executable(${f} ${f}.cpp)
   target_link_libraries(${f} viplnet ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)
endforeach()

