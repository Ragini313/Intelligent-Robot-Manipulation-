#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from moveit_msgs.msg import RobotTrajectory
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from cube_msgs.msg import Cube

cube_positions = []

def cube_position_callback(msg):

    """
    Callback function to process incoming Cube messages and store their poses.
    """

    global cube_positions
    rospy.loginfo(f"Received Cube message - Position: {msg.position}, Orientation: {msg.orientation}")
    
    cube_pose = Pose()
    cube_pose.position = msg.position
    cube_pose.orientation = msg.orientation
    cube_positions.append(cube_pose)
    

def move_to_pose(move_group, pose):
    """
    Move the robot arm to the specified pose.
    """
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
    """
    Main function to stack objects based on received Cube positions.
    """
    global cube_positions
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('stacking_objects', anonymous=True)

    # Initialize MoveIt Commander
    robot = moveit_commander.RobotCommander()
    scene = PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")  # Replace with your robot arm name
    gripper_group = moveit_commander.MoveGroupCommander("panda_hand")  # Replace with your gripper group name
    
    rospy.Subscriber('/cube_positions', Cube, cube_position_callback)

    rospy.loginfo("Waiting for cube positions...")
    
    # Define poses for the stacked objects ... poses are just random for now! Need to be updated
    stack_poses = [
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.2), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.3), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    ]
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        if not cube_positions:
            rospy.loginfo("No cube positions received yet. Waiting...")
            rate.sleep()
            continue

    rospy.loginfo("Processing received cube positions...")
    
    # Pick and place cubes
    """for pose in stack_poses:
        # Move to object picking position
        success = move_to_pose(move_group, pose)
        if success:
            pick_object(move_group, gripper_group)  # Pick the object
            pose.position.z += 0.1
            place_object(move_group, gripper_group, pose)  # Place the object in the new location
    
    moveit_commander.roscpp_shutdown()"""
    for i, cube_pose in enumerate(cube_positions):
            if i >= len(stack_poses):
                rospy.logwarn("No more stack positions available!")
                break

            # 获取目标堆叠位置
            target_pose = stack_poses[i]
            rospy.loginfo(f"Moving Cube to stack position {i + 1}: {target_pose}")

            # 移动到 Cube 的抓取位置
            if move_to_pose(move_group, cube_pose):
                pick_object(move_group, gripper_group)

                # 移动到目标堆叠位置并放置
                place_object(move_group, gripper_group, target_pose)

        # 清空已处理的 Cube 列表
        cube_positions.clear()
        rospy.loginfo("Completed processing all cube positions.")

    moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    try:
        stack_objects()
    except rospy.ROSInterruptException:
        pass

