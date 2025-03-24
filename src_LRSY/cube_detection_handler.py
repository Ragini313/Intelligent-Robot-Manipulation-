#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose

from typing import List


class CubeHandler(object):
    def __init__(self, num_cubes: int = 15):
        rospy.loginfo("Init cube handler")
        self._num_cubes = num_cubes

        self._cube_poses: List[Pose] = []
        self._idx = 0

        self._get_cube_poses()

    def get_poses(self):
        return self._cube_poses


    def _get_point_cloud(self):
        for cube_id in range(self._num_cubes):
            subscriber = rospy.Subscriber(f"cube_{cube_id}_pointcloud", Pose, self._point_cloud_callback)



    def _point_cloud_callback(self, msg):
        ...

    
    def _pose_callback(self, msg: Pose):
        self._cube_poses.append(msg)


    def _get_cube_poses(self):
        for cube_id in range(self._num_cubes):
            subscriber = rospy.Subscriber(f"cube_{cube_id}_pose", Pose, self._pose_callback)
            rospy.wait_for_message(f"cube_{cube_id}_pose", Pose)
            subscriber.unregister()


    def get_current_cube_pose(self) -> Pose:
        cube_pose: Pose = self._cube_poses[self._idx]
        self._idx += 1
        rospy.loginfo(f"Next cube pose to move to: {cube_pose}")
        return cube_pose