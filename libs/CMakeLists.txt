add_library(filesystem filesystem.cpp)
target_include_directories(filesystem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(cam_config cam_config.cpp)
target_include_directories(cam_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cam_config filesystem)

add_library(multicam_mapper multicam_mapper.cpp)
target_include_directories(multicam_mapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
