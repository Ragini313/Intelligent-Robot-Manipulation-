#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose
from moveit_commander.robot_trajectory import RobotTrajectory
from moveit_commander.planning_interface import PlanningSceneInterface

def move_to_pose(move_group, pose):
    move_group.set_pose_target(pose)
    success = move_group.go(wait=True)
    return success

def pick_object(move_group, gripper_group):
    # Assuming a simple "close" gripper function
    gripper_group.set_named_target("close")
    gripper_group.go(wait=True)

def place_object(move_group, gripper_group, pose):
    # Assuming a simple "open" gripper function
    gripper_group.set_named_target("open")
    gripper_group.go(wait=True)
    
    # Move to desired place pose
    move_to_pose(move_group, pose)

def stack_objects():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('stacking_objects', anonymous=True)

    # Initialize MoveIt Commander
    robot = moveit_commander.RobotCommander()
    scene = PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")  # Replace with your robot arm name
    gripper_group = moveit_commander.MoveGroupCommander("panda_hand")  # Replace with your gripper group name

    # Define poses for the stacked objects ... poses are just random for now! Need to be updated
    stack_poses = [
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.2), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.3), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    ]

    # Pick and place cubes
    for pose in stack_poses:
        # Move to object picking position
        success = move_to_pose(move_group, pose)
        if success:
            pick_object(move_group, gripper_group)  # Pick the object
            place_object(move_group, gripper_group, pose)  # Place the object in the new location

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    stack_objects()

