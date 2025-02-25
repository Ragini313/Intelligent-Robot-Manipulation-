#!/usr/bin/python3
import rospy
import sys
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_matrix, quaternion_from_matrix,quaternion_multiply
import numpy as np
from moveit_msgs.msg import Grasp
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

class PickCommander:
    def __init__(self, object_name: str): # , cube_pose: PoseStamped
        """Initialize MoveIt components"""
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Pick Commander", anonymous=True)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("panda_arm")

        rospy.loginfo("Ready setup")
        self.object_name = object_name
        # self.cube_pose = cube_pose
        self.scene_pub = rospy.Publisher("/move_group/planning_scene", PlanningScene, queue_size=10)
        rospy.sleep(1.0)

    def add_object_to_scene(self, cube_pose):
        """Adds an object to the MoveIt planning scene using CollisionObject."""
        rospy.loginfo(f"Adding object '{self.object_name}' to the scene...")

        # **Check if the object already exists**
        existing_objects = self.scene.get_known_object_names()
        if self.object_name in existing_objects:
            rospy.logwarn(f"Object '{self.object_name}' already exists. Removing it first.")
            self.scene.remove_world_object(self.object_name)
            rospy.sleep(1.0)

        # **Add the object to the scene**
        self.scene.add_box(self.object_name, cube_pose, size=(0.05, 0.05, 0.05))

        # **Wait for the object to be successfully added**
        rospy.sleep(1.0)  # Wait for the scene to update
        max_retries = 10
        while self.object_name not in self.scene.get_known_object_names() and max_retries > 0:
            rospy.sleep(0.2)
            max_retries -= 1

        # **Confirm if the object was successfully added**
        if self.object_name in self.scene.get_known_object_names():
            rospy.loginfo(f"Successfully added '{self.object_name}' to the planning scene!")
        else:
            rospy.logwarn(f"Failed to add '{self.object_name}' to the scene after multiple retries.")

    def release_object(self):
        """Releases and removes the currently attached object from the robot"""
        attached_objects = self.scene.get_attached_objects()
        if attached_objects:
            rospy.loginfo(f"Releasing attached objects: {list(attached_objects.keys())}")
            for obj_name in attached_objects.keys():
                rospy.loginfo(f"Removing attached object: {obj_name}")
                self.scene.remove_attached_object("panda_link8", obj_name)
                rospy.sleep(0.5)
        rospy.sleep(1.0)  # Wait for the scene to update

    def adjust_gripper_pose(self, cube_pose: PoseStamped) -> PoseStamped:
        """Adjusts the gripper pose to align with the top of the cube"""
        pose_tmp = PoseStamped()
        pose_tmp.header = cube_pose.header
        pose_tmp.pose = cube_pose.pose

        # Extract the rotation matrix
        q = pose_tmp.pose.orientation
        cube_quaternion = [q.x, q.y, q.z, q.w]
        cube_rot_matrix = quaternion_matrix(cube_quaternion)[:3, :3]

        # Calculate local axes
        local_axes = {
            "top": cube_rot_matrix[:, 2],
            "bottom": -cube_rot_matrix[:, 2],
            "front": cube_rot_matrix[:, 1],
            "back": -cube_rot_matrix[:, 1],
            "left": cube_rot_matrix[:, 0],
            "right": -cube_rot_matrix[:, 0],
        }

        # Find the top face whose normal is closest to the world Z-axis
        world_z = np.array([0, 0, 1])
        top_face = max(local_axes, key=lambda axis: np.dot(local_axes[axis], world_z))
        top_normal = local_axes[top_face]

        rospy.loginfo(f"Determined top face direction: {top_face}")

        # Align the gripper's -Z axis with the top face normal
        gripper_z_axis = -top_normal

        # Choose a reference vector
        reference_vector = np.array([1, 0, 0]) if abs(np.dot(world_z, gripper_z_axis)) > 0.99 else world_z

        # Calculate the gripper coordinate system
        gripper_x_axis = np.cross(reference_vector, gripper_z_axis)
        gripper_x_axis /= np.linalg.norm(gripper_x_axis)
        gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)

        # Generate the rotation matrix and quaternion
        gripper_rot_matrix = np.column_stack((gripper_x_axis, gripper_y_axis, gripper_z_axis))
        gripper_quaternion = quaternion_from_matrix(
            np.vstack((np.column_stack((gripper_rot_matrix, [0, 0, 0])), [0, 0, 0, 1]))
        )
        rotation_90_z = [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
        rotated_quaternion = quaternion_multiply(gripper_quaternion, rotation_90_z)

        adjusted_gripper_pose = PoseStamped()
        adjusted_gripper_pose.header.frame_id = "panda_link0"
        adjusted_gripper_pose.pose.position = pose_tmp.pose.position
        adjusted_gripper_pose.pose.orientation.x = rotated_quaternion[0]
        adjusted_gripper_pose.pose.orientation.y = rotated_quaternion[1]
        adjusted_gripper_pose.pose.orientation.z = rotated_quaternion[2]
        adjusted_gripper_pose.pose.orientation.w = rotated_quaternion[3]

        rospy.loginfo(f"Adjusted gripper_pose:\n{adjusted_gripper_pose}")
        return adjusted_gripper_pose

    def create_grasp(self, gripper_pose: PoseStamped) -> list:
        """Creates a MoveIt-compatible Grasp object for picking an object"""
        grasp = Grasp()

        # **Adjust gripper pose to align with the object**
        adjusted_pose = gripper_pose
        grasp.grasp_pose.header.frame_id = "panda_link0"

        grasp.grasp_pose.pose.position.x = adjusted_pose.pose.position.x
        grasp.grasp_pose.pose.position.y = adjusted_pose.pose.position.y
        grasp.grasp_pose.pose.position.z = adjusted_pose.pose.position.z + 0.1
        grasp.grasp_pose.pose.orientation = adjusted_pose.pose.orientation

        # **Pre-grasp approach (move down towards the object)**
        grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
        grasp.pre_grasp_approach.direction.vector.z = -1.0  # Move down in Z
        grasp.pre_grasp_approach.min_distance = 0.07  # Keep a small safety margin
        grasp.pre_grasp_approach.desired_distance = 0.12  # Approach from further away

        # **Open gripper before grasping**
        grasp.pre_grasp_posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        pre_grasp_point = JointTrajectoryPoint()
        pre_grasp_point.positions = [0.04, 0.04]  # Open both fingers fully
        pre_grasp_point.time_from_start = rospy.Duration(1.0)
        grasp.pre_grasp_posture.points.append(pre_grasp_point)

        # **Grasp action (closing the gripper)**
        grasp.grasp_posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        grasp_point = JointTrajectoryPoint()
        grasp_point.positions = [0.001, 0.001]  # Close fully
        grasp_point.effort = [60.0, 60.0]  # Stronger grip force
        grasp_point.time_from_start = rospy.Duration(1.0)
        grasp.grasp_posture.points.append(grasp_point)

        # **Lift object after grasping**
        grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
        grasp.post_grasp_retreat.direction.vector.z = 1.0  # Move up
        grasp.post_grasp_retreat.min_distance = 0.1  # Ensure a safe lift
        grasp.post_grasp_retreat.desired_distance = 0.15  # Move further away after lift

        return [grasp]

    def execute_pick(self, pose):
        """Executes the pick operation using MoveIt grasping"""
        rospy.sleep(1.0)
        rospy.loginfo("Starting pick operation...")

        # **Ensure no objects are attached before grasping**
        self.release_object()
        rospy.sleep(1.0)

        # **Adjust the gripper pose to align with the object**
        gripper_pose = self.adjust_gripper_pose(pose)
        rospy.sleep(1.0)

        # **Add the object to the scene**
        self.add_object_to_scene(cube_pose=pose)

        # **Generate grasp configurations**
        grasps = self.create_grasp(gripper_pose)

        # **Execute the grasp using MoveIt**
        success = self.group.pick(self.object_name, grasps)

        if success:
            rospy.loginfo("Grasp successful!")
            rospy.sleep(1.0)
        else:
            rospy.logwarn("Grasp failed! Retrying...")




def main():
    import tf.transformations as tft
    cube_pose = PoseStamped()

    cube_pose.header.frame_id = "panda_link0"
    position = [0.6, 0.2, 0.13]
    angles = [0.0, 0.0, 1.4]
    quats = tft.quaternion_from_euler(*angles)
    cube_pose.pose.position.x = position[0]
    cube_pose.pose.position.y = position[1]
    cube_pose.pose.position.z = position[2]

    cube_pose.pose.orientation.x = quats[0]
    cube_pose.pose.orientation.y = quats[1]
    cube_pose.pose.orientation.z = quats[2]
    cube_pose.pose.orientation.w = quats[3]

    picker = PickCommander("cube_1")
    picker.execute_pick(cube_pose)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted. Exiting...")