add_executable(find_solution find_solution.cpp)
target_link_libraries(find_solution cam_config multicam_mapper aruco ${pcl_libs} ${OpenCV_LIBS})

add_executable(detect_markers detect_markers.cpp)
target_link_libraries(detect_markers multicam_mapper cam_config aruco ${pcl_libs} ${OpenCV_LIBS})
