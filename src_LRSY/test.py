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
from pyramid import pyramid_poses_3_tier, pyramid_poses_5_tier, pyramid_poses_1_tier, pyramid_poses_2_tier, pyramid_poses_4_tier


def build_pyramid(iterations):
    robot_mover = RobotMover()
    quats = transformations.quaternion_from_euler(math.pi, 0, math.pi / 4)

    start_pose = Pose()
    start_pose.position.x = 0.4
    start_pose.position.y = -0.2
    start_pose.position.z = 0.2
    start_pose.orientation.x = quats[0]
    start_pose.orientation.y = quats[1]
    start_pose.orientation.z = quats[2]
    start_pose.orientation.w = quats[3]

    target_pose = Pose()
    target_pose.position.x = 0.75
    target_pose.position.y = 0.05
    target_pose.position.z = 0.2
    target_pose.orientation.x = quats[0]
    target_pose.orientation.y = quats[1]
    target_pose.orientation.z = quats[2]
    target_pose.orientation.w = quats[3]

    second_level_pose = Pose()
    second_level_pose.position.x = 0.75
    second_level_pose.position.y = 0.05
    second_level_pose.position.z = 0.2
    second_level_pose.orientation.x = quats[0]
    second_level_pose.orientation.y = quats[1]
    second_level_pose.orientation.z = quats[2]
    second_level_pose.orientation.w = quats[3]

    for i in range(iterations):
        robot_mover.move_arm_to_pose(pose=start_pose)
        rospy.sleep(2)
        robot_mover.move_down(0.07)
        rospy.sleep(3)
        robot_mover.close_gripper()
        rospy.sleep(1)
        robot_mover.move_up()
        rospy.sleep(1)
        target_pose.position.y += -0.06 if i != 0 else 0
        target_pose.position.z = 0.2
        robot_mover.move_arm_to_pose(pose=target_pose)
        rospy.sleep(1)
        robot_mover.move_down(0.06)
        rospy.sleep(1)
        robot_mover.open_gripper()
        rospy.sleep(1)
        robot_mover.move_up()
        rospy.sleep(1)
        robot_mover.move_to_ready_pose()
        rospy.sleep(1)

    robot_mover.move_arm_to_pose(pose=start_pose)
    rospy.sleep(2)
    robot_mover.move_down(0.07)
    rospy.sleep(3)
    robot_mover.close_gripper()
    rospy.sleep(1)
    robot_mover.move_up()
    rospy.sleep(1)
    second_level_pose.position.y = 0.02
    second_level_pose.position.z = 0.25
    robot_mover.move_arm_to_pose(pose=second_level_pose)
    rospy.sleep(1)
    robot_mover.move_down(0.065)
    rospy.sleep(1)
    robot_mover.open_gripper()
    rospy.sleep(1)
    robot_mover.move_up()
    rospy.sleep(1)
    robot_mover.move_to_ready_pose()
    rospy.sleep(1)
    return


def pose_callback(msg: Pose):
    global cube_poses
    cube_pose = Pose()
    cube_pose.position = msg.position
    cube_pose.orientation = msg.orientation
    cube_poses.append(cube_pose)

def apply_offset_picking(y_coordinate):
    y_coordinate_abs = abs(y_coordinate)
    offset = 0
    if 0 < y_coordinate_abs < 0.1:
        offset = 0.005
    elif 0.1 <= y_coordinate_abs < 0.2:
        offset = 0.01
    elif 0.2 <= y_coordinate_abs < 0.3:
        offset = 0.015

    offset = offset if y_coordinate > 0 else -offset
    return offset

cube_poses = []


def make_tower_coords(num_cubes):
    z = 0.2
    positions = [[0.73, -0.2, z + (0.045*i)] for i in range(0, num_cubes, 1)]
    return positions



def test(num_cubes, tier):

    if tier == 1:
        pyramid_poses = make_tower_coords(num_cubes=num_cubes)
    elif tier == 2:
        pyramid_poses = pyramid_poses_2_tier
    elif tier == 3:
        pyramid_poses = pyramid_poses_3_tier
    elif tier == 4:
        pyramid_poses = pyramid_poses_4_tier
    else:
        pyramid_poses = pyramid_poses_5_tier

    print(pyramid_poses)
    robot_mover = RobotMover()
    rospy.sleep(2)
    print(num_cubes)
    robot_mover._move_arm_to_observation_pose()
    rospy.sleep(1)
    for i in range(num_cubes):
        print(i)
        sub = rospy.Subscriber(f"/cube_{i}_pose", Pose, pose_callback)
        rospy.wait_for_message(f"/cube_{i}_pose", Pose)
        sub.unregister()
    rate = rospy.Rate(10)
    global cube_poses
    quats = transformations.quaternion_from_euler(math.pi, 0, math.pi / 4)
    robot_mover.move_to_ready_pose()
    idx = 2
    # start_pose = Pose()
    # start_pose.position.x = cube_poses[idx].position.x
    # start_pose.position.y = cube_poses[idx].position.y
    # start_pose.position.z = cube_poses[idx].position.z
    # start_pose.orientation.x = quats[0]
    # start_pose.orientation.y = quats[1]
    # start_pose.orientation.z = quats[2]
    # start_pose.orientation.w = quats[3]

    # target_pose = Pose()
    # target_pose.position.x = pyramid_poses[idx][0]
    # target_pose.position.y = pyramid_poses[idx][1]
    # target_pose.position.z = pyramid_poses[idx][2]
    # target_pose.orientation.x = quats[0]
    # target_pose.orientation.y = quats[1]
    # target_pose.orientation.z = quats[2]
    # target_pose.orientation.w = quats[3]

    for i in range(num_cubes):
        start_pose = Pose()
        start_pose = cube_poses[i]
        start_pose.position.x += 0.04 #cube_poses[i].position.x
        # start_pose.position.y += apply_offset_picking(y_coordinate=start_pose.position.y) #cube_poses[i].position.y
        start_pose.position.z -= 0.1 #cube_poses[i].position.z
        # start_pose.orientation.x = quats[0]
        # start_pose.orientation.y = quats[1]
        # start_pose.orientation.z = quats[2] # np.deg2rad(-15)
        # start_pose.orientation.w = quats[3]
        robot_mover.move_to_pregrasp(start_pose)
        rospy.sleep(1)
        robot_mover.move_down(0.06)
        rospy.sleep(1)
        robot_mover.picker.grasp()
        rospy.sleep(1)
        robot_mover.move_to_ready_pose()
        rospy.sleep(1)


        target_pose = Pose()
        target_pose.position.x = pyramid_poses[i][0]
        target_pose.position.y = pyramid_poses[i][1]
        target_pose.position.z = pyramid_poses[i][2]
        target_pose.orientation.x = quats[0]
        target_pose.orientation.y = quats[1]
        target_pose.orientation.z = quats[2]
        target_pose.orientation.w = quats[3]
        robot_mover.move_arm_to_pose(pose=target_pose)
        # robot_mover.move_to_pregrasp(start_pose)
        rospy.sleep(1)
        
        robot_mover.move_down(0.055)
        rospy.sleep(1)
        robot_mover.open_gripper()
        # robot_mover.close_gripper()
        # rospy.sleep(2)
        # robot_mover.move_up()
        # robot_mover.close_gripper()
        # rospy.sleep(1)
        robot_mover.move_to_ready_pose()
        rospy.sleep(1)
        
        # target_pose.position.z = 0.2 + (i * 0.06)
        # robot_mover.move_arm_to_pose(pose=target_pose)
        # robot_mover.move_to_pregrasp(target_pose, offset=0.0)
        # robot_mover.move_arm_to_pose(target_pose)
        # rospy.sleep(1)
        # robot_mover.move_down(0.06)
        # rospy.sleep(1)
        # robot_mover.open_gripper()
        # rospy.sleep(1)
        # robot_mover.move_up()
        # rospy.sleep(1)
        # robot_mover.move_to_ready_pose()
        # rospy.sleep(1)


def callback_joint_states(msg):
    print(msg)
    rospy.spin()


def main():
    num_cubes = -1
    tier = -1
    while tier < 0 and num_cubes < 0:
        tier = int(input("Give tier of pyramid: "))

        if tier == 1:
            num_cubes = int(input("Give number of cubes to stack: "))
        elif tier == 2:
            num_cubes = 3
        elif tier == 3:
            num_cubes = 6
        elif tier == 4:
            num_cubes = 10
        elif tier == 5:
            num_cubes = 15
        else:
            print("Given tier is not defined")

    test(num_cubes=num_cubes, tier=tier)


# [-0.0003141398948124972, -0.7846943862227076, 0.00017600309895691389,
# -2.3567236798899063, -0.0002444393433233702, 1.5709534139573171, 0.785163377982711,
# Finger Joints
#  0.0349225178360939, 0.0349225178360939]

## /zed2/zed_node/left_raw/image_raw_color
## /zed2/zed_node/depth/depth_registered


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
