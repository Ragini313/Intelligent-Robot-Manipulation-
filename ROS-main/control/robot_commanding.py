#!/usr/bin/env python3

import sys
import math
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import tf.transformations as transformations
from geometry_msgs.msg import Pose
import numpy as np
from control.robot_mover import RobotMover
from typing import List
import tf.transformations as tft
from lars_perception.cube_detection_handler import CubeHandler


class RobotCommander(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node("Robot_Commander", anonymous=True)

        self.robot_mover = RobotMover()

        # self.voice_commander = ...
        # self.perception_controller = CubeHandler(num_cubes=3)
        self.pyramid_points = None
        self.idx = 0


        rospy.loginfo("Robot Commander initialized")



    def _pick_cube(self, cube_pose: Pose) -> bool:
        self.robot_mover.move_to_pregrasp(cube_pose=cube_pose)
        rospy.sleep(1)
        self.robot_mover.move_down()
        rospy.sleep(1)
        self.robot_mover.close_gripper()
        rospy.sleep(1)


    def _place_cube(self, target_pose: Pose) -> bool:
        self.robot_mover.move_to_pose(target_pose)
        rospy.sleep(1)
        self.robot_mover.move_down()
        rospy.sleep(1)
        self.robot_mover.open_gripper()
        


    def run(self, cube_poses: List[Pose] = None, num_cubes: int = 1) -> None: # List only for testing
        i = 0
        num_cubes_to_pick = len(cube_poses) if cube_poses is not None else num_cubes
        self.robot_mover.open_gripper()
        while i < num_cubes_to_pick:
            cube_pose = cube_poses[i] # else cube_poses[i] self.perception_controller.get_current_cube_pose() if not 
            self._pick_cube(cube_pose=cube_pose)
            motion_sucess = self.robot_mover.move_to_ready_pose()
            rospy.sleep(1)
            self._place_cube(target_pose=self.pyramid_points[i])
            motion_sucess = self.robot_mover.move_to_ready_pose()
            rospy.sleep(1)
            i += 1

                                

    def reset(self) -> bool:
        success = self.robot_mover.move_to_ready_pose()
        self.robot_mover.open_gripper()
        return success
    
    def update_pyramid_pose(self) -> None:
        rospy.logerr("update_pyramid_pose currently not mplemented")
        rospy.sleep(1)


    def test_update_pose(self, target_poses: List[Pose]) -> None:
        self.pyramid_points = target_poses
        rospy.sleep(1)



def get_pre_grasp_pose(x, y, z, rotation) -> Pose:
    pose = Pose()
    quats = tft.quaternion_from_euler(0, 0, rotation)
    pose = Pose()
    pose.orientation.x = quats[0]
    pose.orientation.y = quats[1]
    pose.orientation.z = quats[2]
    pose.orientation.w = quats[3]

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    return pose

def get_target_pose(x, y, z) -> Pose:

    quats = tft.quaternion_from_euler(0, 0, 0)
    pose = Pose()
    pose.orientation.x = quats[0]
    pose.orientation.y = quats[1]
    pose.orientation.z = quats[2]
    pose.orientation.w = quats[3]

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    return pose


def build_pre_grasp_poses(pre_grasp_positions: List[Pose]) -> List[Pose]:
    pre_grasp_poses = []
    for pre_grasp_position in pre_grasp_positions:
        pre_grasp_pose = get_pre_grasp_pose(*pre_grasp_position)
        pre_grasp_poses.append(pre_grasp_pose)

    return pre_grasp_poses


def build_target_poses(target_positions: List[Pose]) -> List[Pose]:
    target_poses = []
    for target_position in target_positions:
        pre_grasp_pose = get_target_pose(*target_position)
        target_poses.append(pre_grasp_pose)

    return target_poses


def control_main():
    robot_commander = RobotCommander()
    # (x, y, z)
    target_positions = [(0.5, -0.1, 0.2), (0.5, -0.1, 0.25), (0.5, -0.1, 0.3)]
    # (x, y, z, orientation)
    pre_grasp_positions = [(0.4, 0.1, 0.1, 0.978), (0.6, 0.2, 0.1, 1.1), (0.7, 0.14, 0.1, 0.0)]

    target_poses = build_target_poses(target_positions)
    # print(f"Target Poses: \n{target_poses}")
    pre_grasp_poses = build_pre_grasp_poses(pre_grasp_positions)
    # print(f"Pre Grasp Poses: \n{pre_grasp_poses}")

    test_pose = Pose()
    test_pose.position.x = 0.4
    test_pose.position.y = 0.1
    test_pose.position.z = 0.1

    # quats = robot_commander.robot_mover.test_conv_ori(63)
    # print(quats)
    quats = tft.quaternion_from_euler(0, 0, 1.1)
    test_pose.orientation.x = quats[0]
    test_pose.orientation.y = quats[1]
    test_pose.orientation.z = quats[2]
    test_pose.orientation.w = quats[3]
    # print(quats)
    j1 = 0.004425608632595956 
    j2 = -0.1776332457239861
    j3 = -0.04997807565715949
    j4 = -1.825997224179561
    j5 = -0.0032382293592747883
    j6 = 1.0
    j7 = 0.783730496518573
    joint_states = [j1, j2, j3, j4, j5, j6, j7]
    # robot_commander.reset()
    robot_commander.robot_mover.close_gripper()
    # robot_commander.robot_mover.move_to_pregrasp(test_pose)
    # robot_commander.robot_mover.open_gripper()
    
    # robot_commander.robot_mover.move_to_pose_by_joints(joint_states)
    # current_pose = robot_commander.robot_mover.arm_group.get_current_pose()
    # current_pose.pose.position.x = 0.58
    # current_pose.pose.position.y = 0.13
    # robot_commander.robot_mover.move_to_observation_pose(current_pose)

    # robot_commander.test_update_pose(target_poses=target_poses)
    # perception_controller = CubeHandler(num_cubes=3)
    # cubes = perception_controller.get_poses()
    # print(cubes)
    # robot_commander.run(cubes, num_cubes=3)
