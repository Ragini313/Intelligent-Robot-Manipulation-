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



class RobotMover(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Robot_Mover", anonymous=True)

        self.picker = PickController()


        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("panda_hand")
        self.robot = moveit_commander.RobotCommander()

        self.planning_frame = self.robot.get_planning_frame()
        self.ready_pose: Pose = self.arm_group.get_current_pose().pose
        rospy.loginfo(f"Ready Pose: {self.ready_pose}")

        self.cube_poses = []
        self.pre_grasp_z_offset = 0.10

        rospy.loginfo("RobotMover initialized and waiting...")


    def move_to_ready_pose(self):
        self.arm_group.set_pose_target(self.ready_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if success:
            rospy.loginfo("Robot arm back in ready pose")
        else:
            rospy.logerr("Robot arm was not able to go back to ready pose")



    def move_to_pregrasp(self, cube_pose: Pose, offset=0.10):
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
        
        T_offset = transformations.translation_matrix([0,0, offset])

        R_corr = transformations.quaternion_matrix(
                    transformations.quaternion_from_euler(math.pi, 0, math.pi/4))
        
        # Combine the offset and the correction rotation.
        T_offset_final = np.dot(T_offset, R_corr)

        T_target = np.dot(T_cube, T_offset_final)

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
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()


    def open_gripper(self):
        rospy.loginfo("Opening gripper...")
        self.gripper_group.set_named_target("open")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()


    def move_arm_to_pose(self, pose: Pose = None):
        rospy.loginfo("Move to cube pose")
        self.arm_group.set_pose_target(pose)
        self.arm_group.go(wait=True)
        self.arm_group.stop()             
        self.arm_group.clear_pose_targets()


    def move_to_poses(self, cube_position, target_position):
        cube_pose = Pose()
        target_pose = Pose()

        quats = transformations.quaternion_from_euler(math.pi, 0, math.pi / 4)
        cube_pose.position.x = cube_position[0]
        cube_pose.position.y = cube_position[1]
        cube_pose.position.z = cube_position[2]

        cube_pose.orientation.x = quats[0]
        cube_pose.orientation.y = quats[1]
        cube_pose.orientation.z = quats[2]
        cube_pose.orientation.w = quats[3]

        target_pose.position.x = target_position[0]
        target_pose.position.y = target_position[1]
        target_pose.position.z = target_position[2]

        target_pose.orientation.x = quats[0]
        target_pose.orientation.y = quats[1]
        target_pose.orientation.z = quats[2]
        target_pose.orientation.w = quats[3]

        rospy.loginfo("Move to cube pose")
        self.arm_group.set_pose_target(cube_pose)
        self.arm_group.go(wait=True)
        self.arm_group.stop()             
        self.arm_group.clear_pose_targets()
        rospy.loginfo("Reached start pose with offset")
        rospy.sleep(1)

        self.open_gripper()

        rospy.loginfo("Moving down")
        self.move_down(0.06)
        rospy.sleep(1)

        # self.picker.grasp()
        rospy.sleep(1)

        rospy.loginfo("Moving up")
        # self.move_up(0.1)

        # rospy.loginfo("Move to target pose")
        # self.arm_group.set_pose_target(target_pose)
        # self.arm_group.go(wait=True)
        # self.arm_group.stop()             
        # self.arm_group.clear_pose_targets()

        # rospy.loginfo("Movement done")
        # self.move_down(0.06)

        # self.open_gripper()




    def move_up(self, lift_distance=0.1):
        current_pose = self.arm_group.get_current_pose().pose
        target_pose = Pose()
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z + lift_distance
        target_pose.orientation = current_pose.orientation

        self.arm_group.set_pose_target(target_pose)
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()



    def move_down(self, z_position = 0.10):
        target_pose = self.arm_group.get_current_pose().pose
        target_pose.position.z -= z_position
        self.arm_group.set_pose_target(target_pose)
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
