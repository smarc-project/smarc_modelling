cmake_minimum_required(VERSION 3.8)
project(piml_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find_package(<dependency> REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(smarc_msgs REQUIRED)
find_package(sam_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ament_cmake REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SynchedData.msg"
  "msg/ThrusterRPMStamped.msg"
  DEPENDENCIES rosgraph_msgs sensor_msgs smarc_msgs nav_msgs sam_msgs std_msgs
 )
 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
