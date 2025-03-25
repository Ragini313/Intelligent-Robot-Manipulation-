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
from sensor_msgs.msg import JointState


# , [x, -0.02, z_0], [x, -0.08, z_0]
# , [x, 0.005, z_1], [x, -0.055, z_1]
x = 0.75
z_0 = 0.2
z_1 = 0.245
z_2 = 0.29
z_3 = 0.335
z_4 = 0.38

pyramid_poses_1_tier = [[x, -0.2, z_0], \
                        [x, -0.2, z_1], \
                        [x, -0.2, z_2], \
                        [x, -0.2, z_3], \
                        [x, -0.2, z_4]]


pyramid_poses_2_tier = [[x, 0.16, z_0], [x, 0.10, z_0],
                 [x, 0.13, z_1]]

pyramid_poses_3_tier = [[x, 0.16, z_0], [x, 0.10, z_0], [x, 0.04, z_0], \
                 [x, 0.13, z_1], [x, 0.07, z_1], \
                 [x, 0.1, z_2]] 

pyramid_poses_4_tier = [[x, 0.16, z_0], [x, 0.10, z_0], [x, 0.04, z_0], [x, -0.02, z_0], \
                 [x, 0.13, z_1], [x, 0.07, z_1], [x, 0.01, z_1], \
                 [x, 0.1, z_2], [x, 0.04, z_2], \
                 [x, 0.07, z_3]] 

pyramid_poses_5_tier = [[x, 0.16, z_0], [x, 0.10, z_0], [x, 0.04, z_0], [x, -0.02, z_0], [x, -0.08, z_0], \
                 [x, 0.13, z_1], [x, 0.07, z_1], [x, 0.01, z_1], [x, -0.05, z_1], \
                 [x, 0.1, z_2], [x, 0.04, z_2], [x, -0.02, z_2], \
                 [x, 0.07, z_3], [x, 0.01, z_3], \
                 [x, 0.04, z_4]] 


