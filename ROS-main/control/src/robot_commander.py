#!/usr/bin/env python3

import sys
import math
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import tf.transformations as transformations
from geometry_msgs.msg import Pose
import numpy as np
from robot_controller import RobotMover
from typing import List


class RobotCommander(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Robot_Commander", anonymous=True)

        self.robot_mover = RobotMover()

        # self.voice_commander = ...
        # self.perception_controller = ...
        self.pyramid_points = None
        self.idx = 0


        rospy.loginfo("Robot Commander initialized")


    def run(self, cube_poses: List[Pose]) -> None:
        """
        Runs the overall cube stacking algorithm
        Assumes a list of poses given for the whole current layer
        """
        is_ready_pose: bool = self.robot_mover.is_in_ready_pose()
        if self.pyramid_points is None:
            raise TypeError("Pyramid Points is None, call update method first")

        i = 0
        while i < len(cube_poses):
            cube_pose = cube_poses[i]
            if is_ready_pose:
                pre_grasp_motion_success: bool = self.robot_mover.move_to_pregrasp(cube_pose=cube_pose)
                if pre_grasp_motion_success:
                    rospy.sleep(1)
                    move_down_success: bool = self.robot_mover.move_down()
                    if move_down_success:
                        rospy.sleep(1)
                        grasp_success: bool = self.robot_mover.close_gripper()
                        if grasp_success:
                            rospy.sleep(1)
                            move_up_success: bool = self.robot_mover.move_up()
                            if move_up_success:
                                rospy.sleep(1)
                                motion_sucess = self.robot_mover.move_to_pose(self.pyramid_points[self.idx])
                                if motion_sucess:
                                    i += 1
                                    self.update_pyramid_pose()
        
            self.robot_mover.open_gripper()
            rospy.sleep(1)
            is_ready_pose = self.robot_mover.move_to_ready_pose()

    
    def update_pyramid_pose(self) -> None:
        rospy.logerr("update_pyramid_pose currently not mplemented")
        rospy.sleep(1)


    def test_update_pose(self, target_poses: List[Pose]) -> None:
        self.pyramid_points = target_poses
        rospy.sleep(1)