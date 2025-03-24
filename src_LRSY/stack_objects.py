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

from moveit_msgs.msg import Grasp, GripperTranslation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import PlaceLocation

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

## To get the pose from ZED2_pose_Estimator:

    

def move_to_pose(move_group, pose):
    """
    Move the robot arm to the specified pose.
    """
    move_group.set_pose_target(pose)
    success = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    return success

def pick_object(move_group, gripper_group, scene, cube_pose, cube_id):
    """
    Uses MoveIt's built-in grasp planning for picking.
    """

    rospy.loginfo("Generating grasp message for picking.")

    # Define the grasp message
    grasp = Grasp()
    grasp.id = cube_id  # Assign a unique ID to the grasp action
    grasp.grasp_pose.header.frame_id = "panda_link0"  # Adjust for your robot's frame
    grasp.grasp_pose.pose = cube_pose  # Use detected cube pose

    # Pre-grasp approach: Move towards the object before grasping
    grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
    grasp.pre_grasp_approach.direction.vector.x = 1.0  # Move forward towards the object
    grasp.pre_grasp_approach.min_distance = 0.05  # Minimum approach distance
    grasp.pre_grasp_approach.desired_distance = 0.1  # Desired approach distance

    # Gripper joint position for pre-grasp (open)
    grasp.pre_grasp_posture.joint_names = ["panda_finger_joint1"]
    pre_grasp_point = JointTrajectoryPoint()
    pre_grasp_point.positions = [0.04]  # Open fingers
    pre_grasp_point.time_from_start = rospy.Duration(0.5)
    grasp.pre_grasp_posture.points.append(pre_grasp_point)

    # Gripper joint position for grasp (close)
    grasp.grasp_posture.joint_names = ["panda_finger_joint1"]
    grasp_point = JointTrajectoryPoint()
    grasp_point.positions = [0.00]  # Close fingers
    grasp_point.time_from_start = rospy.Duration(0.5)
    grasp.grasp_posture.points.append(grasp_point)

    # Lift after grasping
    grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
    grasp.post_grasp_retreat.direction.vector.z = 1.0  # Move up
    grasp.post_grasp_retreat.min_distance = 0.05
    grasp.post_grasp_retreat.desired_distance = 0.1

    rospy.loginfo("Attempting to pick the object.")
    success = move_group.pick(cube_id, [grasp])

    if success:
        rospy.loginfo("Pick successful!")
    else:
        rospy.logwarn("Pick failed.")

    return success
    

def place_object(move_group, scene, cube_id, target_pose):
    """
    Uses MoveIt's built-in place planning for placing.
    """

    rospy.loginfo("Generating place message.")

    # Define place message
    place_location = PlaceLocation()
    place_location.id = cube_id
    place_location.place_pose.header.frame_id = "panda_link0"
    place_location.place_pose.pose = target_pose  # Target pose for stacking

    # Approach before placing
    place_location.pre_place_approach.direction.header.frame_id = "panda_link0"
    place_location.pre_place_approach.direction.vector.z = -1.0  # Move down
    place_location.pre_place_approach.min_distance = 0.05
    place_location.pre_place_approach.desired_distance = 0.1

    # Retreat after placing
    place_location.post_place_retreat.direction.header.frame_id = "panda_link0"
    place_location.post_place_retreat.direction.vector.z = 1.0  # Move up
    place_location.post_place_retreat.min_distance = 0.05
    place_location.post_place_retreat.desired_distance = 0.1

    # Open gripper to release object
    place_location.post_place_posture.joint_names = ["panda_finger_joint1"]
    open_point = JointTrajectoryPoint()
    open_point.positions = [0.04]  # Open fingers
    open_point.time_from_start = rospy.Duration(0.5)
    place_location.post_place_posture.points.append(open_point)

    rospy.loginfo("Attempting to place the object.")
    success = move_group.place(cube_id, [place_location])

    if success:
        rospy.loginfo("Place successful!")
    else:
        rospy.logwarn("Place failed.")

    return success


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
    move_group = moveit_commander.MoveGroupCommander("panda_arm")
    gripper_group = moveit_commander.MoveGroupCommander("panda_hand")
    
    rospy.Subscriber('/cube_positions', Cube, cube_position_callback)

    rospy.loginfo("Waiting for cube positions...")
    rospy.sleep(2)  # Allow time for positions to update

    stack_poses = [
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.15), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
        Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.2), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    ]

    ### code below is to line them up next to each other along the x-axis, instead of stacking them
    # stack_poses = [
    #     Pose(position=geometry_msgs.msg.Point(0.5, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    #     Pose(position=geometry_msgs.msg.Point(0.55, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    #     Pose(position=geometry_msgs.msg.Point(0.6, 0.0, 0.1), orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)),
    # ]

    if not cube_positions:
        rospy.loginfo("No cube positions received yet. Waiting...")
        return

    rospy.loginfo("Processing received cube positions...")

    for i, cube_pose in enumerate(cube_positions):
        if i >= len(stack_poses):
            rospy.logwarn("No more stack positions available!")
            break

        cube_id = f"cube_{i}"  # Unique ID for each cube
        scene.add_box(cube_id, cube_pose, (0.05, 0.05, 0.05))  # Add cube to planning scene

        rospy.loginfo(f"Picking Cube {i + 1}")
        if pick_object(move_group, gripper_group, scene, cube_pose, cube_id):
            rospy.sleep(1)

            rospy.loginfo(f"Placing Cube {i + 1}")
            place_object(move_group, scene, cube_id, stack_poses[i])
        
    cube_positions.clear()
    rospy.loginfo("Completed processing all cube positions.")

    moveit_commander.roscpp_shutdown()



if __name__ == '__main__':
    try:
        rospy.loginfo("Starting stacking_objects node...")
        stack_objects()
    except rospy.ROSInterruptException:
        rospy.logwarn("ROS Interrupt received. Shutting down...")
    finally:
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt! and ROS shutdown complete.")


