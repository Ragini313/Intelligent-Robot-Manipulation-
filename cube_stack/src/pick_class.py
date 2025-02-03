#!/usr/bin/python3
import rospy
import sys
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander

from geometry_msgs.msg import PoseStamped
from cube_msgs.msg import Cube
from moveit_msgs.msg import Grasp, GripperTranslation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np


class Pick:
    def __init__(self, object_name: str, cube_pose: PoseStamped):
        # 初始化 MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("panda_arm")

        self.group.set_planner_id("RRTConnectkConfigDefault") 
        self.group.set_planning_time(10.0)  # 设置最大规划时间为 10 秒
        self.group.set_num_planning_attempts(10) 
        

        # 设置对象名称和位姿
        self.object_name = object_name
        self.cube_pose = cube_pose

    def release_object(self):
        """释放并删除当前抓取的物体"""
        # 获取所有附着的物体
        attached_objects = self.scene.get_attached_objects()
        rospy.loginfo(f"Attached objects before release: {attached_objects}")
        if attached_objects:
            for attached_name in attached_objects.keys():
                rospy.loginfo(f"Releasing attached object: {attached_name}")
                self.scene.remove_attached_object("panda_link8", attached_name)
                rospy.sleep(1.0)  # 等待同步

        # 获取场景中所有已知物体
        known_objects = self.scene.get_known_object_names()
        for obj_name in known_objects:
            rospy.loginfo(f"Removing object from scene: {obj_name}")
            self.scene.remove_world_object(obj_name)
            rospy.sleep(1.0)  # 等待同步
    def collect_diagnostics(self):
        """收集诊断信息"""
        # 获取当前状态和场景信息
        known_objects = self.scene.get_known_object_names()
        attached_objects = self.scene.get_attached_objects()
        current_pose = self.group.get_current_pose()
        current_state = self.robot.get_current_state()

    
    def clear_scene_objects(self):
        """清空场景中的所有物体"""
        known_objects = self.scene.get_known_object_names()
        while known_objects:
            rospy.loginfo(f"Removing objects: {known_objects}")
            for obj_name in known_objects:
                self.scene.remove_world_object(obj_name)
            rospy.sleep(1.0)  # 确保场景同步
            known_objects = self.scene.get_known_object_names()
        rospy.loginfo("Scene cleared.")


    def add_cube_to_scene(self):
        """将 Cube 添加到 MoveIt 规划场景"""
        rospy.loginfo(f"Adding cube '{self.object_name}' to the scene.")
        cube_size = [0.045, 0.045, 0.045]
        self.scene.add_box(self.object_name, self.cube_pose, cube_size)
        rospy.sleep(1.0)  # 等待场景同步
        rospy.loginfo(f"Cube '{self.object_name}' added to the scene.")



    def adjust_gripper_pose(self) -> PoseStamped:

        """调整抓手姿态，使其与立方体顶部对齐"""
        cube_pose = self.cube_pose
        # 从立方体的姿态中提取旋转矩阵
        q = cube_pose.pose.orientation
        cube_quaternion = [q.x, q.y, q.z, q.w]
        cube_rot_matrix = quaternion_matrix(cube_quaternion)[:3, :3]

        # 定义局部坐标轴
        local_axes = {
            "top": cube_rot_matrix[:, 2],       # Z 轴
            "bottom": -cube_rot_matrix[:, 2],   # -Z 轴
            "front": cube_rot_matrix[:, 1],     # Y 轴
            "back": -cube_rot_matrix[:, 1],     # -Y 轴
            "left": cube_rot_matrix[:, 0],      # X 轴
            "right": -cube_rot_matrix[:, 0],    # -X 轴
        }

        # 找到最接近世界 Z 轴的顶面法线
        world_z = np.array([0, 0, 1])
        top_face = max(local_axes, key=lambda axis: np.dot(local_axes[axis], world_z))
        top_normal = local_axes[top_face]

        rospy.loginfo(f"Determined top face: {top_face}")

        # 将抓手的 -Z 轴与立方体的顶面法线对齐
        gripper_z_axis = -top_normal

        # 选择一个与 Z 轴不平行的参考向量
        reference_vector = np.array([1, 0, 0]) if abs(np.dot(world_z, gripper_z_axis)) > 0.99 else world_z

        # 计算抓手的 X 轴，使其与参考向量和 Z 轴成直角
        gripper_x_axis = np.cross(reference_vector, gripper_z_axis)
        gripper_x_axis /= np.linalg.norm(gripper_x_axis)

        # 计算 Y 轴
        gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)

        # 构建旋转矩阵
        gripper_rot_matrix = np.column_stack((gripper_x_axis, gripper_y_axis, gripper_z_axis))

        # 将旋转矩阵转换为四元数
        gripper_quaternion = quaternion_from_matrix(
            np.vstack((np.column_stack((gripper_rot_matrix, [0, 0, 0])), [0, 0, 0, 1]))
        )

        # 设置并返回调整后的抓手姿态
        gripper_pose = PoseStamped()
        gripper_pose.header.frame_id = "panda_link0"
        gripper_pose.pose.position = cube_pose.pose.position
        gripper_pose.pose.orientation.x = gripper_quaternion[0]
        gripper_pose.pose.orientation.y = gripper_quaternion[1]
        gripper_pose.pose.orientation.z = gripper_quaternion[2]
        gripper_pose.pose.orientation.w = gripper_quaternion[3]

        return gripper_pose

    def create_grasp(self, gripper_pose: PoseStamped) -> list:
        """创建 Grasp 对象"""
        grasp = Grasp()

        # 设置抓取姿态
        grasp.grasp_pose.header.frame_id = gripper_pose.header.frame_id
        grasp.grasp_pose.pose = gripper_pose.pose
        grasp.grasp_pose.pose.position.z -= 0.1
        grasp.allowed_touch_objects = [self.object_name]

        # 预抓取接近运动
        grasp.pre_grasp_approach = GripperTranslation()
        grasp.pre_grasp_approach.direction.header.frame_id = gripper_pose.header.frame_id
        grasp.pre_grasp_approach.direction.vector.z = -1.0
        grasp.pre_grasp_approach.min_distance = 0.1
        grasp.pre_grasp_approach.desired_distance = 0.2

        # 抓取后撤退运动
        grasp.post_grasp_retreat = GripperTranslation()
        grasp.post_grasp_retreat.direction.header.frame_id = gripper_pose.header.frame_id
        grasp.post_grasp_retreat.direction.vector.z = 1.0
        grasp.post_grasp_retreat.min_distance = 0.1
        grasp.post_grasp_retreat.desired_distance = 0.2

        # 设置抓取姿态
        grasp.pre_grasp_posture = JointTrajectory()
        grasp.pre_grasp_posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        pre_grasp_point = JointTrajectoryPoint()
        pre_grasp_point.positions = [0.03, 0.03]
        pre_grasp_point.time_from_start = rospy.Duration(1.0)
        grasp.pre_grasp_posture.points = [pre_grasp_point]

        grasp.grasp_posture = JointTrajectory()
        grasp.grasp_posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        grasp_point = JointTrajectoryPoint()
        grasp_point.positions = [0.01, 0.01]
        grasp_point.effort = [100.0, 100.0]
        grasp_point.time_from_start = rospy.Duration(2.0)
        grasp.grasp_posture.points = [grasp_point]

        return [grasp]

    def move_above_cube(self, gripper_pose: PoseStamped):
        """移动到物体上方"""
        above_pose = PoseStamped()
        above_pose.header.frame_id = "panda_link0"
        above_pose.pose.position = gripper_pose.pose.position
        above_pose.pose.position.z += 0.2 # 移到物体上方
        above_pose.pose.orientation = gripper_pose.pose.orientation

        rospy.loginfo("Moving to position above the cube.")
        self.group.set_pose_target(above_pose)
        success = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if not success:
            rospy.logwarn("Failed to move above the cube.")
    
    def execute_pick(self):
        """执行抓取操作"""
        self.release_object()
        self.clear_scene_objects()
        known_objects = self.scene.get_known_object_names()
        attached_objects = self.scene.get_attached_objects()
        rospy.loginfo(f"Known objects before adding new cube: {known_objects}")
        rospy.loginfo(f"Attached objects before adding new cube: {attached_objects}")

        # 将 Cube 添加到场景
        self.add_cube_to_scene()

        # 调整抓取姿态并执行夹取
        gripper_pose = self.adjust_gripper_pose()
        self.move_above_cube(gripper_pose)
        grasps = self.create_grasp(gripper_pose)

        rospy.loginfo(f"Attempting to execute grasp on '{self.object_name}'...")
        success = self.group.pick(self.object_name, grasps)

        if success:
            rospy.loginfo("Grasp executed successfully.")
            self.collect_diagnostics()
        else:
            rospy.logwarn("Grasp failed.")
            self.collect_diagnostics()