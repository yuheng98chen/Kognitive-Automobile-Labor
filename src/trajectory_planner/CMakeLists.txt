cmake_minimum_required(VERSION 3.0.2)
project(trajectory_planner)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/trajectory_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
