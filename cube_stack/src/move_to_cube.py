#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from stack_objects import cube_para  # Import cube data from stack_objects module

def move_to_cube_position(move_group, cube_data):
    """
    Move the robot arm to the specified cube position.
    """
    target_pose = Pose()
    target_pose.position.x = cube_data["position"]["x"]
    target_pose.position.y = cube_data["position"]["y"]
    target_pose.position.z = cube_data["position"]["z"] + 0.1  # Raise slightly to avoid collisions
    target_pose.orientation.x = cube_data["orientation"]["x"]
    target_pose.orientation.y = cube_data["orientation"]["y"]
    target_pose.orientation.z = cube_data["orientation"]["z"]
    target_pose.orientation.w = cube_data["orientation"]["w"]

    move_group.set_pose_target(target_pose)
    success = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    if success:
        rospy.loginfo(f"Successfully moved to cube ID {cube_data['id']} at position {cube_data['position']}.")
    else:
        rospy.logerr(f"Failed to move to cube ID {cube_data['id']} at position {cube_data['position']}.")

def main():
    rospy.init_node('move_to_cubes_positions', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")  # Replace with your robot arm group name

    rospy.loginfo("Moving to cube positions...")

    for cube_data in cube_para:  # Iterate over cube positions
        print(cube_data)
        move_to_cube_position(move_group, cube_data)

    rospy.loginfo("All cubes have been processed.")
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
