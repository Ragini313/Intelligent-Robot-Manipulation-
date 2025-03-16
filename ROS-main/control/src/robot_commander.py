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


class RobotCommander(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Robot_Commander", anonymous=True)

        self.robot_mover = RobotMover()

        # self.voice_commander = ...
        # self.perception_controller = ...
        self.pyramid_points = []
        self.idx = 0


        rospy.loginfo("Robot Commander initialized")


    def run(self, cube_pose: Pose) -> None:
        is_ready_pose: bool = self.robot_mover.is_in_ready_pose()

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
                                self.update_pyramid_pose()
        
        self.robot_mover.move_to_ready_pose()


    
    def update_pyramid_pose(self) -> None:
        rospy.logerr("update_pyramid_pose currently not mplemented")