#!/usr/bin/python3
import sys
import os
import rospy
from cube_msgs.msg import Cube
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from geometry_msgs.msg import PoseStamped
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from pick_class import Pick

def get_cube_message():
    """从 ROS Topic '/cube' 中接收 Cube 消息"""
    try:
        rospy.loginfo("Waiting for Cube message from /cube topic...")
        cube_msg = rospy.wait_for_message("/cube", Cube, timeout=10.0)
        rospy.loginfo("Received Cube message.")
        return cube_msg
    except rospy.ROSException:
        rospy.logerr("Timeout while waiting for Cube message.")
        return None

def main():
    rospy.init_node('cube_pick_demo')

    # 从 topic 获取 Cube 信息
    cube_msg = get_cube_message()
    if not cube_msg:
        rospy.logerr("Failed to get Cube message. Exiting...")
        return

    # 构建 Cube 位姿
    cube_pose = PoseStamped()
    cube_pose.header.frame_id = "world"
    cube_pose.pose.position = cube_msg.position
    cube_pose.pose.orientation = cube_msg.orientation

    # 创建 Pick 对象并执行抓取
    pick_controller = Pick(object_name=cube_msg.name, cube_pose=cube_pose)
    pick_controller.release_object()
    pick_controller.clear_scene_objects()
    pick_controller.execute_pick()


if __name__ == "__main__":
    try:
    
        main()
    except rospy.ROSInterruptException:
        pass