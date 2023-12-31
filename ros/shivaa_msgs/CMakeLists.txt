cmake_minimum_required(VERSION 3.5)
project(shivaa_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/KinematicsStatus.msg"
  "msg/StateVector.msg"
  "msg/LaneState.msg"
)

set(srv_files
  "srv/GetForwardKinematics.srv"
  "srv/GetInverseKinematics.srv" 
)

set(action_files
  "action/LaneDetection.action"
  "action/FollowPath.action"
)

rosidl_generate_interfaces(${PROJECT_NAME} 
  ${srv_files} ${msg_files} ${action_files} 
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs  
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

