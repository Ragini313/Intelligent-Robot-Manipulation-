#!/usr/bin/env python3

import sys
import rospy
from moveit_commander import MoveGroupCommander, JointState
import moveit_commander
from franka_gripper.msg import GraspActionGoal, GraspActionResult, MoveActionGoal, MoveActionResult

class PickController(object):
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node("Gripper_Controller", anonymous=True)

        self.gripper_move_result = MoveActionResult()
        self.gripper_move_result_subscriber = rospy.Subscriber("/franka_gripper/move/result", MoveActionResult, self.move_result_callback)
        self.gripper_move_publisher = rospy.Publisher("franka_gripper/move/goal", MoveActionGoal, queue_size=10)

        self.gripper_grasp_result = GraspActionResult()
        self.gripper_grasp_result_subscriber = rospy.Subscriber("/franka_gripper/grasp/result", GraspActionResult, self.grasp_result_callback)
        self.gripper_grasp_publisher = rospy.Publisher("franka_gripper/grasp/goal", GraspActionGoal, queue_size=10)

        self.joint_state_subscriber = rospy.Subscriber("/franka_gripper/joint_states", JointState, self.joint_state_callback)
        self.joint_states = JointState()

        rospy.sleep(2)
        rospy.loginfo("Pick Controller initialized")

    def move_result_callback(self, result_msg) -> None:
        self.gripper_moveresult = result_msg


    def grasp_result_callback(self, result_msg) -> None:
        self.gripper_grasp_result = result_msg


    def joint_state_callback(self, result_msg) -> None:
        self.joint_states = result_msg


    def set_width(self, width, speed=0.15):
        move_action_goal = MoveActionGoal()

        move_action_goal.goal.width = width
        move_action_goal.goal.speed = speed

        self.gripper_move_publisher.publish(move_action_goal)
        rospy.sleep(1)
        rospy.loginfo(self.gripper_move_result.result.success)
        return self.gripper_move_result.result.success
    

    def grasp(self):
        grasp_action_goal = GraspActionGoal()
        grasp_action_goal.goal.width = 0.045

        grasp_action_goal.goal.epsilon.inner = 0.05
        grasp_action_goal.goal.epsilon.outer = 0.05

        grasp_action_goal.goal.speed = 0.15
        grasp_action_goal.goal.force = 20.0 # in Newton

        self.gripper_grasp_publisher.publish(grasp_action_goal)
        rospy.sleep(1)