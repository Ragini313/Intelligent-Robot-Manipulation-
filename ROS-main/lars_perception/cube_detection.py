#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose

from tf import TransformListener
import tf.transformations as tft
from cv_bridge import CvBridge

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from lars_perception.point_cloud_handler import PointCloudHandler




class Cube(object):
    def __init__(self, id: int, x: float, y: float, z: float = 0.1, rotation: float = 0.0):
        self.cube_id: int = id
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.rotation: float = rotation


class CubeDetector(object):
    def __init__(self, simulation = False):
        self.BLUR_KERNEL = (27, 27) if not simulation else (23, 23)
        self.TABLE_COLOR = 50 if not simulation else 70

        self._cubes: List[Cube] = list()

        self._image = None
        self._camera_matrix: np.ndarray = None
        self._camera_frame = None
        self._depth_image = None

        self._tf_listener: TransformListener = TransformListener()
        self._cv_bridge: CvBridge = CvBridge()

        self._world_frame: str = "world"

        self._image_subscriber = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self._image_callback)
        self._camera_info_subscriber = rospy.Subscriber("/zed2/zed_node/left/camera_info", CameraInfo, self._camera_info_callback)
        self._depth_image_subscriber = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self._depth_callback)


        self._debug_image_publish = rospy.Publisher("/cube_detection/debug_image", Image, queue_size=10)


    def _depth_callback(self, msg: Image) -> None:
        try:
            self._depth_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            rospy.logerr(e)


    def _camera_info_callback(self, msg: CameraInfo) -> None:
        try:
            self._camera_matrix = np.array(msg.K).reshape(3,3)
            self._camera_frame = msg.header.frame_id
        except Exception as e:
            rospy.logerr(e)


    def _image_callback(self, msg: Image) -> None:
        self._image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        self.run()


    def _preprocess_image(self):
        
        gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, self.BLUR_KERNEL, 0)

        # filter out table
        _, thresholded_image = cv2.threshold(blurred_image, self.TABLE_COLOR, 255, cv2.THRESH_BINARY)
        # image = self._to_polygone(thresholded_image)
        return thresholded_image



    def _divide_two_cubes(self, center_x, center_y, width, height, angle):
        angle_radians = np.radians(angle)
        direction_vector = np.array([np.cos(-angle_radians), np.sin(-angle_radians)])
        move_dist = height / 4
        if (width > height):
            move_dist = width / 4

        center_left = np.array([center_x, center_y]) - direction_vector * move_dist
        center_right = np.array([center_x, center_y]) + direction_vector * move_dist

        left_depth = self._depth_image[center_left[1].astype(int), center_left[0].astype(int)]
        right_depth = self._depth_image[center_right[1].astype(int), center_right[0].astype(int)]


        transformed_left_point = self._transform_pixels_into_camera_frame(center_left[0].astype(int),
                                                                           center_left[1].astype(int),
                                                                           depth=left_depth)
        transformed_right_point = self._transform_pixels_into_camera_frame(center_right[0].astype(int),
                                                                           center_right[1].astype(int),
                                                                           depth=right_depth)

        transformed_point_left = self._transform_cube_point(transformed_left_point)
        transformed_point_right = self._transform_cube_point(transformed_right_point)

        return transformed_point_left, transformed_point_right


    def _to_polygone(self, image):
        mask = np.zeros_like(image)
        polygon = np.array([[150, 100], [500, 110], [480, 300], [180, 300]])
        cv2.fillPoly(mask, [polygon], 255)
        region_of_interest = cv2.bitwise_and(image, mask)

        return region_of_interest
    

    def _get_depth(self, contour) -> float:
        mask = np.zeros(self._depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        depth_values = self._depth_image[mask == 255]
        depth_values = depth_values[(depth_values > 0)]
        if depth_values.size == 0:
            return None
        return np.median(depth_values)


    def _find_cubes(self, contours: List[np.ndarray]) -> List[Cube]:
        found_cubes = list()
        num_cubes = 0
        print(len(contours))
        for contour in contours:
            eps = 0.01 * cv2.arcLength(contour, True)
            edges = cv2.approxPolyDP(contour, eps, True)
            cube_area = abs(cv2.contourArea(contour))
            convex = cv2.isContourConvex(edges)
            if cube_area > 1000: # 500
                moments = cv2.moments(edges)

                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])

                depth = self._depth_image[centroid_y, centroid_x]
                # depth = self._get_depth(contour)

                camera_frame = self._transform_pixels_into_camera_frame(centroid_x, centroid_y, depth)
                transformed_point = self._transform_cube_point(camera_frame)
                # print(f"Transformed Point: {transformed_point}")

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])

                if width < height:
                    angle = 90 - angle
                else:
                    angle = -angle

                if convex:
                    detected_cube = Cube(num_cubes, transformed_point.point.x, transformed_point.point.y, rotation=angle)
                    found_cubes.append(detected_cube)
                    num_cubes += 1
                    # if cube_area < 2000:
                    #     detected_cube = Cube(num_cubes, transformed_point.point.x, transformed_point.point.y, rotation=angle)
                    #     found_cubes.append(detected_cube)
                    #     num_cubes += 1
                    # else:
                    #     cube_left, cube_right = self._divide_two_cubes(centroid_x, centroid_y, width=width, height=height, angle=angle)
                    #     detected_cube = Cube(num_cubes, cube_left.point.x, cube_left.point.y, rotation=angle)
                    #     found_cubes.append(detected_cube)
                    #     num_cubes += 1
                    #     detected_cube = Cube(num_cubes, cube_right.point.x, cube_right.point.y, rotation=angle)
                    #     found_cubes.append(detected_cube)
                    #     num_cubes += 1
        return found_cubes
            


    def _transform_pixels_into_camera_frame(self, x, y, depth):
        homogenous_point = np.array([x, y, 1.0])
        camera_matrix_inverse = np.linalg.inv(self._camera_matrix)
        return np.dot(camera_matrix_inverse, homogenous_point) * depth


    def _transform_cube_point(self, point) -> PointStamped:
        try:
            transformed_point = PointStamped()
            transformed_point.header.frame_id = self._camera_frame
            transformed_point.header.stamp = rospy.Time()
            transformed_point.point.x = point[0]
            transformed_point.point.y = point[1]
            transformed_point.point.z = point[2]
            transformed_point = self._tf_listener.transformPoint(self._world_frame, transformed_point)
            return transformed_point
        except Exception as e:
            rospy.logerr(e)
            return None


    def _publish_cubes(self) -> None:
        if self._cubes:
            for cube in self._cubes:
                cube_pose = Pose()
                cube_pose.position.x = cube.x + 0.02
                cube_pose.position.y = cube.y + 0.02
                cube_pose.position.z = cube.z

                angle = math.radians(cube.rotation)
                quats = tft.quaternion_from_euler(0, 0, angle)
                cube_pose.orientation.x = quats[0]
                cube_pose.orientation.y = quats[1]
                cube_pose.orientation.z = quats[2]
                cube_pose.orientation.w = quats[3]

                rospy.loginfo(f"CubePose: {cube_pose}")
                cube_pose_publisher = rospy.Publisher(f"cube_{cube.cube_id}_pose", Pose, queue_size=10)
                cube_pose_publisher.publish(cube_pose)


    def _publish_debug_image(self, image):
        try:
            debug_image_msg = self._cv_bridge.cv2_to_imgmsg(image)
            self._debug_image_publish.publish(debug_image_msg)
        except Exception as e:
            rospy.logerr(e)


    def show_mage(self, image):
        plt.imshow(image)
        plt.show()


    def run(self):
        if self._image is not None and self._depth_image is not None:
            try:
                preprocessed_image = self._preprocess_image()
                self._publish_debug_image(preprocessed_image)

                detected_edges = cv2.Canny(preprocessed_image, 50, 150)
                # self._publish_debug_image(detected_edges)

                contours, _ = cv2.findContours(detected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self._cubes = self._find_cubes(contours)

                self._publish_cubes()
            
            except Exception as e:
                rospy.logerr(e)


def test_func():
    CubeDetector(simulation=True)
    # PointCloudHandler()
    print("Hello")
    rospy.spin()
