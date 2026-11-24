cmake_minimum_required(VERSION 3.0.2)
project(kal_env_map)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  message_generation
)

add_service_files(
  FILES
  AddStaticObject.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs nav_msgs message_runtime )

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/env_map_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
