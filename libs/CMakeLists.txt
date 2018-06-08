add_library(filesystem filesystem.cpp)

add_library(cam_config cam_config.cpp)
target_link_libraries(cam_config filesystem)

add_library(multicam_mapper multicam_mapper.cpp)

add_library(dataset dataset.cpp)

add_library(image_array_detector image_array_detector.cpp)

add_library(initializer initializer.cpp)
