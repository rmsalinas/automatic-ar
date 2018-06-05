project(automatic-ar)
cmake_minimum_required(VERSION 2.8)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
endif()

#set(CMAKE_PREFIX_PATH "~/local/") #uncomment if you have local libraries compiled in ~/local/

find_package(OpenCV REQUIRED)
find_package(aruco REQUIRED)
find_package(PCL)

if(PCL_FOUND)
    INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
    set(pcl_libs "${PCL_VISUALIZATION_LIBRARIES}")
    add_definitions(-DPCL)
endif(PCL_FOUND)

include_directories(libs)
include_directories(eigen3)
add_subdirectory(libs)
add_subdirectory(apps)












