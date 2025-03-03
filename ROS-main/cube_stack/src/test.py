#!/usr/bin/env python3

import sys
import math
import rospy
import moveit_commander
import tf.transformations as transformations
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry
from cube_msgs.msg import Cube
import numpy as np
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from gripper_controller import PickController
import copy
from robot_mover import RobotMover


def test():
    robot_mover = RobotMover()
    rospy.sleep(2)

    quats = transformations.quaternion_from_euler(math.pi, 0, math.pi / 4)

    start_pose = Pose()
    start_pose.position.x = 0.4
    start_pose.position.y = -0.2
    start_pose.position.z = 0.3 
    start_pose.orientation.x = quats[0]
    start_pose.orientation.y = quats[1]
    start_pose.orientation.z = quats[2]
    start_pose.orientation.w = quats[3]

    robot_mover.move_arm_to_pose(pose=start_pose)
    rospy.sleep(1)
    robot_mover.move_to_ready_pose()




def main():
    test()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass