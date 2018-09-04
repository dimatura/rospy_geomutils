cmake_minimum_required(VERSION 2.8.3)
project(rospy_geomutils)

find_package(catkin REQUIRED COMPONENTS
    rospy
    tf
    tf2
    geometry_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rospy_geomutils
   CATKIN_DEPENDS rospy tf tf2 geometry_msgs
#  DEPENDS system_lib
)
