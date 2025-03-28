#!/usr/bin/env python3

import sys
import math
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import tf.transformations as transformations
from geometry_msgs.msg import Pose
import numpy as np
from control.grasp_controlling import GraspController
from typing import List


class RobotMover(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        # initialize grasp controller for handling closing the gripper
        self.grasp_controller = GraspController()

        # initialize groups (hand and arm)
        self.scene: PlanningSceneInterface = moveit_commander.PlanningSceneInterface()
        self.arm_group: MoveGroupCommander = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group: MoveGroupCommander = moveit_commander.MoveGroupCommander("panda_hand")
        self.robot: RobotCommander = moveit_commander.RobotCommander()

        self.planning_frame = self.robot.get_planning_frame()
        self.ready_pose_joints: List[float] = [0.00012649076641757517, -0.7837772733111574, -1.231260989431604e-05, -2.358868831264023, -1.4127707235189746e-05, 1.5713110896263007, 0.7853939957771825] # Change to joint values
        # rospy.loginfo(f"Ready Pose: {self.ready_pose}")

        rospy.loginfo("RobotMover initialized and waiting...")


    def is_in_ready_pose(self) -> bool:
        return True if self.arm_group.get_current_joint_values() == self.ready_pose_joints else False


    def move_to_ready_pose(self):
        """
        Moves arm back into ready pose
        """
        print(self.ready_pose_joints)
        success = self.arm_group.go(self.ready_pose_joints, wait=True)
        self.arm_group.stop()

        if success:
            rospy.loginfo("Robot arm back in ready pose")
        else:
            rospy.logerr("Robot arm was not able to go back to ready pose")

        return success
    

    def move_to_pose_by_joints(self, joints):
        """
        Moves arm back into ready pose
        """
        # print(self.ready_pose_joints)
        success = self.arm_group.go(joints, wait=True)
        self.arm_group.stop()

        if success:
            rospy.loginfo("Robot arm back in ready pose")
        else:
            rospy.logerr("Robot arm was not able to go back to ready pose")

        return success


    def move_to_pose(self, pose: Pose) -> bool:
        """
        Moves arm to a pose with a "neutral" orientation
        """
        quats = transformations.quaternion_from_euler(math.pi, 0, math.pi/4)
        target_pose: Pose = pose
        target_pose.orientation.x = quats[0]
        target_pose.orientation.y = quats[1]
        target_pose.orientation.z = quats[2]
        target_pose.orientation.w = quats[3]

        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo(f"Robot arm to given pose:\n {target_pose}")
        else:
            rospy.logerr(f"Robot arm was not able to go to pose:\n {target_pose}")

        return success
    
    def move_to_observation_pose(self, pose: Pose) -> bool:
        """
        Moves arm to a pose with a "neutral" orientation
        """
        target_pose: Pose = pose

        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo(f"Robot arm to given pose:\n {target_pose}")
        else:
            rospy.logerr(f"Robot arm was not able to go to pose:\n {target_pose}")

        return success



    def move_to_pregrasp(self, cube_pose: Pose) -> bool:
        rospy.loginfo("Received cube pose:")
        rospy.loginfo(cube_pose)

        cube_quat = [cube_pose.orientation.x,
                     cube_pose.orientation.y,
                     cube_pose.orientation.z,
                     cube_pose.orientation.w]
        
        T_cube = transformations.quaternion_matrix(cube_quat)
        T_cube[0:3, 3] = [cube_pose.position.x,
                          cube_pose.position.y,
                          cube_pose.position.z]
        
        # assuming offset given from perception
        T_offset = transformations.translation_matrix([0,0, cube_pose.position.z])

        # build "correction" matrix to keep robot arm facing down and add offset to hand by pi/4
        R_corr = transformations.quaternion_matrix(
                    transformations.quaternion_from_euler(math.pi, 0, math.pi/4))
        
        # Combine the offset and the correction rotation.
        T_offset_final = np.dot(T_offset, R_corr)

        T_target = np.dot(T_cube, T_offset_final)

        # build pre-grasp pose
        target_position = T_target[0:3, 3]
        target_quat = transformations.quaternion_from_matrix(T_target)
        target_pose = Pose()
        target_pose.position.x = target_position[0]
        target_pose.position.y = target_position[1]
        target_pose.position.z = target_position[2]
        target_pose.orientation.x = target_quat[0]
        target_pose.orientation.y = target_quat[1]
        target_pose.orientation.z = target_quat[2]
        target_pose.orientation.w = target_quat[3]

        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo("Robot moved successfully to pre-grasp pose")
        else:
            rospy.logerr("Encountered error during pre-grasp motion")
        return success


    def open_gripper(self) -> bool:
        success = self.grasp_controller.move_fingers(0.04, 0.04)
        if success:
            rospy.loginfo("Robot successfully opened gripper")
        else:
            rospy.logerr("Encountered error during opening gripper")

        return success


    def close_gripper(self) -> bool:
        rospy.loginfo("Closing Gripper for Grasping")
        success = self.grasp_controller.grasp()
        if success:
            rospy.loginfo("Closing gripper successfully")
        else:
            rospy.logerr("Encountered error during closing gripper")
        return success


    def move_up(self, lift_distance=0.1) -> bool:
        """
        Moves robot arm by lift_distance up
        """
        current_pose = self.arm_group.get_current_pose().pose
        target_pose = Pose()
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z + lift_distance
        target_pose.orientation = current_pose.orientation

        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo("Robot moved successfully up")
        else:
            rospy.logerr("Encountered error during moving up")
        return success



    def move_down(self, downward_distance = 0.07) -> bool:
        """
        Moves robot arm by lift_distance down
        """
        target_pose = self.arm_group.get_current_pose().pose
        target_pose.position.z -= downward_distance
        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo("Robot moved successfully moved down")
        else:
            rospy.logerr("Encountered error during downward movement")
        return success

