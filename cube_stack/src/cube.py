#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Pose,Quaternion
from cube_msgs.msg import Cube
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix, euler_from_matrix
from franka_gripper.msg import MoveActionGoal
from franka_gripper.msg import GraspActionGoal

# List to store cube data
global cube_para
cube_para = []


def cube_callback(msg):
    """
    Callback function to process incoming Cube messages and store their poses.
    """
    global cube_para
    #rospy.loginfo(f"Received Cube message - Position: {msg.position}, Orientation: {msg.orientation}, Size: {msg.size}")
    cube_data = {
        "id": len(cube_para),
        "position": {
            "x": msg.position.x,
            "y": msg.position.y,
            "z": msg.position.z,
        },
        "orientation": {
            "x": msg.orientation.x,
            "y": msg.orientation.y,
            "z": msg.orientation.z,
            "w": msg.orientation.w,
        },
        "size": msg.size,
    }
    cube_para.append(cube_data)
    #rospy.loginfo(f"Cube data appended for grasping: {cube_data}")



def move_to_target(move_group, cube_data,Quaternion,offset):
    """
    Move the robot arm to the position directly above the cube (position target).
    """
    try:
        # 定义位置目标
        orientation_target = Pose()
        orientation_target.position.x = cube_data["position"]["x"]
        orientation_target.position.y = cube_data["position"]["y"]
        orientation_target.position.z = cube_data["position"]["z"] + offset 

        
        orientation_target.orientation.x =  Quaternion["x"]
        orientation_target.orientation.y =  Quaternion["y"]
        orientation_target.orientation.z =  Quaternion["z"]
        orientation_target.orientation.w =  Quaternion["w"]

        
        # move_group.set_planner_id("RRTConnectkConfigDefault")  # 使用 RRTConnect
        # move_group.set_max_velocity_scaling_factor(0.0.5)       
        # move_group.set_max_acceleration_scaling_factor(0.0.5)   
        move_group.set_pose_target(orientation_target)
        move_group.set_start_state_to_current_state()
        


        
        plan = move_group.plan()
        if plan and plan[0]:
            rospy.loginfo("Executing position target movement...")
            move_group.execute(plan[1], wait=True)
            rospy.loginfo(f"Successfully moved to position target for cube ID {cube_data['id']}.")
        else:
            rospy.logwarn("No valid plan found for position target.")
    except Exception as e:
        rospy.logerr(f"Error during position target movement: {e}")

def open_gripper():
    """
    发布命令到 /franka_gripper/move/goal 话题以张开夹具。
    """
    gripper_pub = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal, queue_size=10)

    rospy.sleep(1)  # 等待发布器初始化

    goal_msg = MoveActionGoal()
    goal_msg.goal.width = 0.08
    goal_msg.goal.speed = 0.1  

    rospy.loginfo(f"Publishing gripper open command with width={goal_msg.goal.width} and speed={goal_msg.goal.speed}...")
    gripper_pub.publish(goal_msg)
    rospy.loginfo("Gripper open command sent.")
    rospy.sleep(2)  

def grasp_object(width=0.01, speed=0.1, force=60):
    
    gripper_pub = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal, queue_size=10)
    rospy.sleep(1)  

    goal_msg = GraspActionGoal()
    goal_msg.goal.width = width  
    goal_msg.goal.speed = speed  
    goal_msg.goal.force = force  
    goal_msg.goal.epsilon.inner = 0.2  
    goal_msg.goal.epsilon.outer = 0.2

    rospy.loginfo(f"Grasping object with width={width}, speed={speed}, force={force}...")
    gripper_pub.publish(goal_msg)
    rospy.sleep(2)  


def compute_upward_face_and_normal(cube_orientation):

    local_normals = {
        "front": np.array([0.0, 0.0, 1.0]),
        "back": np.array([0.0, 0.0, -1.0]),
        "left": np.array([-1.0, 0.0, 0.0]),
        "right": np.array([1.0, 0.0, 0.0]),
        "top": np.array([0.0, 1.0, 0.0]),
        "bottom": np.array([0.0, -1.0, 0.0]),
    }

    rot_matrix = quaternion_matrix([
        cube_orientation["x"],
        cube_orientation["y"],
        cube_orientation["z"],
        cube_orientation["w"]
    ])[:3, :3]

    global_normals = {
        face: np.dot(rot_matrix, normal)
        for face, normal in local_normals.items()
    }

    z_axis = np.array([0.0, 0.0, 1.0])
    face_scores = {face: np.dot(normal, z_axis) for face, normal in global_normals.items()}
    upward_face = max(face_scores, key=face_scores.get)
    upward_normal = global_normals[upward_face]

    return upward_face, upward_normal

def compute_gripper_orientation(upward_normal):

    from tf.transformations import quaternion_from_matrix

    gripper_z_axis = -upward_normal
    reference_vector = np.array([1.0, 0.0, 0.0])  
    if np.abs(np.dot(gripper_z_axis, reference_vector)) > 0.99:  
        reference_vector = np.array([0.0, 1.0, 0.0])  

    
    gripper_x_axis = np.cross(reference_vector, gripper_z_axis)
    gripper_x_axis /= np.linalg.norm(gripper_x_axis)  

    gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)
    gripper_y_axis /= np.linalg.norm(gripper_y_axis)  

    

    
    gripper_rot_matrix = np.column_stack((gripper_x_axis, gripper_y_axis, gripper_z_axis))

    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = gripper_rot_matrix  
    gripper_quaternion = quaternion_from_matrix(homogeneous_matrix)

    return {
        "x": gripper_quaternion[0],
        "y": gripper_quaternion[1],
        "z": gripper_quaternion[2],
        "w": gripper_quaternion[3],
    }



def main():
    """
    Main function to initialize the node and move to cubes sequentially.
    """
    global cube_para

    rospy.init_node('move_to_cubes_positions', anonymous=True)
    
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")

    rospy.Subscriber('cube', Cube, cube_callback)
    rospy.loginfo("Cube subscriber node started. Waiting for cube data...")

    rate = rospy.Rate(10)
    target_data = {
                "id": len(cube_para),
                "position": {
                        "x": 0.4,
                        "y": 0.2,
                        "z": -0.2,
                },
                "orientation": {
                        "x": 1.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 0.0,
                }
            }
    offset1 = 0.3
    offset2 = 0.1

    while not rospy.is_shutdown():
        if cube_para:
            rospy.loginfo("Cube data received. Moving to each cube sequentially...")
            open_gripper()
            rospy.sleep(1)

            
            cube_data = cube_para[0]
            cube_orientation = cube_data['orientation']
            print(cube_orientation)

            upward_face, upward_normal = compute_upward_face_and_normal(cube_orientation)

            print(f"朝上的面: {upward_face}")
            print(f"朝上的法向量: {upward_normal}")

            gripper_orientation = compute_gripper_orientation(upward_normal)
            print(f"夹具目标姿态: {gripper_orientation}")
            move_to_target(move_group,cube_data,gripper_orientation,offset1)
            
            
            
            move_to_target(move_group, cube_data,gripper_orientation, offset2)
            
            grasp_object()

            rospy.sleep(2)
            # 等待夹取操作完成
            move_to_target(move_group,cube_data,gripper_orientation, offset1)
            
            
            
            move_to_target(move_group,target_data, gripper_orientation, offset1)
            #smove_to_target(move_group,target_data,target_data["orientation"], offset2)
            open_gripper()
            
            rospy.loginfo("All cubes have been processed. Exiting...")
            break

        rate.sleep()

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
