#!/usr/bin/env python3

from robot_commander import RobotCommander
import rospy
from geometry_msgs.msg import Pose
import tf.transformations as tft
from typing import List


def get_pre_grasp_pose(x, y, z, rotation) -> Pose:
    pose = Pose()
    quats = tft.quaternion_from_euler(0, 0, rotation)
    pose = Pose()
    pose.orientation.x = quats[0]
    pose.orientation.y = quats[1]
    pose.orientation.z = quats[2]
    pose.orientation.w = quats[3]

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    return pose

def get_target_pose(x, y, z) -> Pose:

    quats = tft.quaternion_from_euler(0, 0, 0)
    pose = Pose()
    pose.orientation.x = quats[0]
    pose.orientation.y = quats[1]
    pose.orientation.z = quats[2]
    pose.orientation.w = quats[3]

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    return pose


def build_pre_grasp_poses(pre_grasp_positions: List[Pose]) -> List[Pose]:
    pre_grasp_poses = []
    for pre_grasp_position in pre_grasp_positions:
        pre_grasp_pose = get_target_pose(*pre_grasp_position)
        pre_grasp_poses.append(pre_grasp_pose)

    return pre_grasp_poses


def build_target_poses(target_positions: List[Pose]) -> List[Pose]:
    target_poses = []
    for target_position in target_positions:
        pre_grasp_pose = get_pre_grasp_pose(*target_position)
        target_poses.append(pre_grasp_pose)

    return target_poses


def main():
    robot_commander = RobotCommander()
    # (x, y, z)
    target_positions = [()]
    # (x, y, z, orientation)
    pre_grasp_positions = [()]

    target_poses = build_target_poses(target_positions)
    print(f"Target Poses: \n{target_poses}")
    pre_grasp_poses = build_pre_grasp_poses(pre_grasp_positions)
    print(f"Pre Grasp Poses: \n{pre_grasp_poses}")

    robot_commander.test_update_pose(target_poses=target_poses)
    robot_commander.run(pre_grasp_poses)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass