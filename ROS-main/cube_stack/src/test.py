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



class RobotMover(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Robot Mover", anonymous=True)


        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("panda_hand")
        self.robot = moveit_commander.RobotCommander()

        self.planning_frame = self.robot.get_planning_frame()

        self.cube_poses = []
        self.pre_grasp_z_offset = 0.00

        rospy.loginfo("RobotMover initialized and waiting...")


    # def add_collision_object(self, object_id, pose: PoseStamped):
    #     collision_object = CollisionObject()
    #     collision_object.header.frame_id = self.planning_frame
    #     collision_object.id = object_id

    #     box = SolidPrimitive()
    #     box.type = SolidPrimitive.BOX
    #     box.dimensions = [0.045, 0.045, 0.045]

    #     collision_object.primitives = [box]
    #     collision_object.primitive_poses = [pose.pose]
    #     collision_object.operation = CollisionObject.ADD

    #     self.scene.add_object(collision_object)

    def add_collision_object(self, cube_pose, object_name):
        self.scene.add_box(object_name, cube_pose, size=(0.05,0.05,0.05))



    def move_to_start_position(self):
        target_pose = Pose()

        target_pose.position.x = 0.5
        target_pose.position.y = 0.0
        target_pose.position.z = 0.4

        quats = transformations.quaternion_from_euler(math.pi,0,0)
        target_pose.orientation.x = quats[0]
        target_pose.orientation.y = quats[1]
        target_pose.orientation.z = quats[2]
        target_pose.orientation.w = quats[3]

        self.arm_group.set_pose_target(target_pose)
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()



    def move_to_pregrasp(self, cube_pose: PoseStamped, offset=0.10):
        rospy.loginfo("Received cube pose:")
        rospy.loginfo(cube_pose)

        cube_quat = [cube_pose.pose.orientation.x,
                     cube_pose.pose.orientation.y,
                     cube_pose.pose.orientation.z,
                     cube_pose.pose.orientation.w]
        
        T_cube = transformations.quaternion_matrix(cube_quat)
        T_cube[0:3, 3] = [cube_pose.pose.position.x,
                          cube_pose.pose.position.y,
                          cube_pose.pose.position.z]
        
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


    # def open_gripper(self):
    #     joint_values = self.hand_group.get_current_joint_values()
    #     joint_values[0] = 0.04
    #     joint_values[1] = 0.04
    #     self.hand_group.set_joint_value_target(joint_values)
    #     success = self.hand_group.go(wait=True)
    #     self.hand_group.stop()
    #     if success:
    #         rospy.loginfo("Gripper opened successfully.")
    #         return True
    #     else:
    #         rospy.logwarn("Failed to open gripper. Check MoveIt logs.")
    #         return False

    def open_gripper(self):
        rospy.loginfo("Opening gripper...")
        self.gripper_group.set_named_target("open")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()


    def close_gripper(self):
        rospy.loginfo("Closing gripper...")
        self.gripper_group.set_named_target("close")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()


    def move_up(self, lift_distance=0.1):
        current_pose = self.arm_group.get_current_pose().pose
        target_pose = Pose()
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z + lift_distance
        target_pose.orientation = current_pose.orientation

        rospy.loginfo("Lifting cube upward by %.2f m.", lift_distance)
        self.arm_group.set_pose_target(target_pose)
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if plan_success:
            rospy.loginfo("Cube lifted successfully.")
        else:
            rospy.logwarn("Failed to lift cube.")


    def attach_object(self, object_id):
        grasping_group = self.gripper_group.get_name()
        touch_links = self.robot.get_link_names(group=grasping_group)
        self.scene.attach_box(self.arm_group.get_end_effector_link(), object_id, touch_links=touch_links)
        rospy.loginfo("Attached object: %s", object_id)
        
        # Allow some time for the planning scene to update.
        rospy.sleep(1)
        
        # Remove the object from the world collision list.
        # self.scene.remove_world_object(object_id)
        rospy.loginfo("Removed %s from world collision objects.", object_id)


    def move_down(self, z_position = 0.10):
        target_pose = self.arm_group.get_current_pose().pose
        target_pose.position.z = z_position
        self.arm_group.set_pose_target(target_pose)
        plan_success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()


    def grasp_cube(self, cube_pose, collision_object_id="cube1"):
        rospy.loginfo("Starting grasp routine for %s...", collision_object_id)
        # Step 1: Add cube as a collision object.
        self.add_collision_object(collision_object_id, cube_pose)
        rospy.sleep(1)

        # Step 2: Open the gripper.
        self.open_gripper()
        rospy.sleep(1)

        # Step 3: Move to pre-grasp (e.g., 10 cm above the cube).
        self.move_to_pregrasp(cube_pose, offset=0.10)
        rospy.sleep(1)

        # Step 4: Move down to grasp pose (e.g., 2 cm above the cube).
        self.move_down()
        rospy.sleep(1)

        # Step 5: Close the gripper.
        self.close_gripper()
        rospy.sleep(1)

        # Step 6: Attach the object to simulate a successful grasp.
        self.attach_object(collision_object_id)
        rospy.sleep(1)


        # Step 7: Lift the cube.
        self.move_up(lift_distance=0.1)

    # def pick_up_cubes(self, cubes):

    #     for cube in cubes:
    #         rospy.loginfo("Move to cube Position: x=%.2f, y=%.2f, z=%.2f" % (cube.position.x, cube.position.y, cube.position.z))
    #         self._move_to_pregrasp_pose(cube)
    #         rospy.sleep(3.0)

    #     moveit_commander.roscpp_shutdown()


def wait_for_state_update(scene, box_name, box_is_known=False, box_is_attached=False, timeout=4):
    """
    Waits for the planning scene to update so that the box is either added or attached.
    This function polls the scene until the object appears (or disappears) as expected.
    """
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (rospy.get_time() - start < timeout) and not rospy.is_shutdown():
        # Check if the box is attached
        attached_objects = scene.get_attached_objects([box_name])
        is_attached = len(attached_objects.keys()) > 0

        # Check if the box is known in the planning scene
        is_known = box_name in scene.get_known_object_names()

        if (box_is_attached == is_attached) and (box_is_known == is_known):
            return True
        rospy.sleep(0.1)
    return False



def main():

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("Robot Mover", anonymous=True)


    scene = moveit_commander.PlanningSceneInterface()
    # arm_group = moveit_commander.MoveGroupCommander("panda_arm")
    arm_mover = RobotMover()

    box_pose = PoseStamped()
    box_pose.header.frame_id = "panda_link0"
    q = transformations.quaternion_from_euler(0.0, 0.0, 1.4)
    position = [0.6, 0.2, 0.13]
    box_pose.pose.orientation.x = q[0]
    box_pose.pose.orientation.y = q[1]
    box_pose.pose.orientation.z = q[2]
    box_pose.pose.orientation.w = q[3]

    box_pose.pose.position.x = position[0]
    box_pose.pose.position.y = position[1]
    box_pose.pose.position.z = position[2]-0.1


    gripper_group = arm_mover.gripper_group
    closed_gripper_joints = [0.04, 0.04]
    # gripper_group.set_joint_value_target(closed_gripper_joints)
    # gripper_group.go(wait=True)
    robot = arm_mover.robot
    touch_links = robot.get_link_names(group="panda_hand")
    print(touch_links)
    eef_link = arm_mover.arm_group.get_end_effector_link()
    print(eef_link)
    # print(box_pose) 
    box_name = "cube1"
    size = 0.045

    # print(box_pose)
    # scene.remove_attached_object(eef_link, name=box_name)
    # scene.remove_world_object(box_name)
    # scene.add_box(box_name, box_pose, size = (size, size, size))

    print(scene.get_objects())

    # arm_mover.move_to_pregrasp(box_pose)
    # arm_mover.open_gripper()
    # arm_mover.move_down(0.13)
    # arm_mover.close_gripper()
    # scene.attach_box(eef_link, box_name, touch_links=touch_links)
    wait_for_state_update(scene, box_name, box_is_attached=True, box_is_known=False)
    arm_mover.move_up(0.3)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass