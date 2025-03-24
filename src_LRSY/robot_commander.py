#!/usr/bin/env python3
import sys
import rospy
from geometry_msgs.msg import PoseStamped
from pick_class import PickCommander
import tf.transformations as tft


class RobotCommander():
    def __init__(self):
        ...


def main():
    cube_pose = PoseStamped()

    cube_pose.header.frame_id = "panda_link0"
    position = [0.6, 0.2, 0.13]
    angles = [0.0, 0.0, 1.4]
    quats = tft.quaternion_from_euler(*angles)
    cube_pose.pose.position.x = position[0]
    cube_pose.pose.position.y = position[1]
    cube_pose.pose.position.z = position[2]

    cube_pose.pose.orientation.x = quats[0]
    cube_pose.pose.orientation.y = quats[1]
    cube_pose.pose.orientation.z = quats[2]
    cube_pose.pose.orientation.w = quats[3]

    picker = PickCommander("cube_1")
    picker.adjust_gripper_pose(cube_pose)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted. Exiting...")