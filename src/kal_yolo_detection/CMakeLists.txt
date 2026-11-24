cmake_minimum_required(VERSION 3.0.2)
project(kal_yolo_detection)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    rospy
    sensor_msgs
    std_msgs
    geometry_msgs
    message_generation
)

add_message_files(
    FILES
    NamedMaskArray.msg
)

generate_messages(
    DEPENDENCIES std_msgs sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs geometry_msgs message_runtime
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
    scripts/yolo_detection_node.py
    scripts/yolo_pt_mask_node.py
    scripts/yolo_onnx_mask_node.py
    scripts/yolo_inference_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY model/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/model
    FILES_MATCHING PATTERN "*.onnx" PATTERN "*.pt" PATTERN "*.pth" PATTERN "*.py"
)
