#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose

from tf import TransformListener
import tf.transformations as tft
from cv_bridge import CvBridge

import cv2
import numpy as np
from typing import List, Tuple


BLUR_KERNEL = (27, 27)
MIN_EDGES = 4
MAX_EDGES = 10
MIN_AREA = 3800
MAX_AREA = 5500
TABLE_COLOR = 50


class Cube(object):
    def __init__(self, id: int, x: float, y: float, z: float = 0.2, rotation: float = 0.0):
        self.cube_id: int = id
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.rotation: float = rotation


class CubeDetector(object):
    def __init__(self):
        self._cubes: List[Cube] = list()

        self._image = None
        self._camera_matrix: np.ndarray = None
        self._depth_image = None

        self._tf_listener: TransformListener = TransformListener()
        self._cv_bridge: CvBridge = CvBridge()

        self._world_frame: str = "world"

        self._image_subscriber = rospy.Subscriber("", Image, self._image_callback)
        self._camera_info_subscriber = rospy.Subscriber("", CameraInfo, self._camera_info_callback)
        self._depth_image_subscriber = rospy.Subscriber("", Image, self._depth_callback)


    def _depth_callback(self, msg: Image) -> None:
        try:
            self._depth_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            rospy.logerr(e)


    def _camera_info_callback(self, msg: CameraInfo) -> None:
        try:
            self._camera_matrix = np.array(msg.K).reshape(3,3)
        except Exception as e:
            rospy.logerr(e)


    def _image_callback(self, msg: Image) -> None:
        self._image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)


    def _preprocess_image(self):
        gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL, 0)

        # filter out table
        _, thresholded_image = cv2.threshold(blurred_image, TABLE_COLOR, 255, cv2.THRESH_BINARY)
        image = self._to_polygone(thresholded_image)
        return image



    def _to_polygone(self, image):
        mask = np.zeros_like(image)
        polygon = np.array([[150, 100], [500, 110], [480, 300], [180, 300]])
        cv2.fillPoly(mask, [polygon], 255)
        region_of_interest = cv2.bitwise_and(image, mask)

        return region_of_interest
    

    def _find_cubes(self, contours: List[np.ndarray]) -> List[Cube]:
        found_cubes = list()
        num_cubes = 0

        for contour in contours:
            eps = 0.01 * cv2.arcLength(contour, True)
            edges = cv2.approxPolyDP(contour, eps, True)
            cube_area = abs(cv2.contourArea(contour))

            if cube_area > 500:
                moments = cv2.moments(edges)

                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])

                depth = self._depth_image[centroid_y, centroid_x]

                camera_frame = self._transform_pixels_into_camera_frame(centroid_x, centroid_y, depth)
                transformed_point = self._transform_cube_point(camera_frame)

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])

                if width < height:
                    angle = 90 - angle
                else:
                    angle *= -1


                detected_cube = Cube(num_cubes, transformed_point.x, transformed_point.y, rotation=angle)
                found_cubes.append(detected_cube)
                num_cubes += 1
        return found_cubes
            


    def _transform_pixels_into_camera_frame(self, x, y, depth):
        homogenous_point = np.array([x, y, 1.0])
        camera_matrix_inverse = np.linalg.inv(self._camera_matrix)
        return np.dot(camera_matrix_inverse, homogenous_point) * depth


    def _transform_cube_point(self, point):
        try:
            transformed_point = Point()
            transformed_point.x = point[0]
            transformed_point.y = point[1]
            transformed_point.z = point[2]
            transformed_point = self._tf_listener.transformPoint(self._world_frame, transformed_point)
            return transformed_point
        except Exception as e:
            rospy.logerr(e)
            return None


    def _publish_cubes(self) -> None:
        if self._cubes:
            for cube in self._cubes:
                cube_pose = Pose()
                cube_pose.position.x = cube.x
                cube_pose.position.y = cube.y
                cube_pose.position.z = cube.z

                quats = tft.quaternion_from_euler(0, 0, cube.rotation)
                cube_pose.orientation.x = quats[0]
                cube_pose.orientation.y = quats[1]
                cube_pose.orientation.z = quats[2]
                cube_pose.orientation.w = quats[3]

                cube_pose_publisher = rospy.Publisher(f"cube_{cube.cube_id}_pose", Pose, queue_size=10)
                cube_pose_publisher.publish(cube_pose)


    def run(self):
        if self._image is not None and self._depth_image is not None:
            try:
                thresholded_image = self._preprocess_image()

                detected_edges = cv2.Canny(thresholded_image, 50, 150)

                contours, _ = cv2.findContours(detected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self._cubes = self._find_cubes(contours)

                self._publish_cubes()
            
            except Exception as e:
                rospy.logerr(e)