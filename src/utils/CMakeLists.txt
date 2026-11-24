cmake_minimum_required(VERSION 3.0.2)
project(utils)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)
find_package(OpenCV REQUIRED)

catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES utils
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(image_undistort_downscale_node src/image_undistort_downscale.cpp)
target_link_libraries(image_undistort_downscale_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS image_undistort_downscale_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/utils/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
