#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf.transformations as tft
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_geometry_msgs


class CubeDetector(object):
    def __init__(self):
        rospy.init_node("cube_odometry_node", anonymous=True)

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.depth_image = None
        self.image = None

        self.camera_matrix = None
        self.dist_coeffs = None

        self.image_subscriber = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        self.depth_subscriber = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.camera_info_subscriber = rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, self.camera_info_callback)

        rospy.loginfo("Cube Odometry Node initialized")
        rospy.spin()

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3,3)
        self.dist_coeffs = np.array(msg.D)
        # self.camera_info_subscriber.unregister()

        rospy.loginfo("Camera Info received")


    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth_image = depth_image
        except Exception as e:
            rospy.logerr("Depth Image conversion failed: {}".format(e))



    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("Image conversion failed: {}".format(e))

        self.process_image(cv_image=cv_image)


    def process_image(self, cv_image):

        image = cv_image.copy()

        transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([10, 50, 50])
        upper_bound = np.array([30, 255, 255])
        mask = cv2.inRange(transformed_image, lower_bound, upper_bound)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            rospy.loginfo("No contours found.")
            return
        
        cube_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(cube_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # rospy.loginfo(f"Rect: {rect}")


        center = (int(rect[0][0]), int(rect[0][1]))
        angle = rect[2]
        # rospy.loginfo(f"Angle: {angle}")

        cv2.drawContours(image, [box], 0, (255, 0,0), 1)
        # cv2.imshow("Contours", image)
        # cv2.waitKey(1)

        depth_value = self.depth_image[center[1], center[0]] if self.depth_image is not None else None
        # rospy.loginfo(f"Depth value: {depth_value}")

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        u, v = center

        Xc = (u - cx) * depth_value / fx
        Yc = (v - cy) * depth_value / fy
        Zc = depth_value

        cube_point_camera = PointStamped()
        cube_point_camera.header.stamp = rospy.Time.now()
        cube_point_camera.header.frame_id = "zed_left_camera_frame"
        cube_point_camera.point.x = Xc
        cube_point_camera.point.y = Yc
        cube_point_camera.point.z = Zc

        # rospy.loginfo(f"Cube Point in Camera {cube_point_camera}")
        point_world = self.transform_into_world(cube_point_camera)


        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"  # World (robot) frame
        pose_msg.pose.position.x = point_world.point.x
        pose_msg.pose.position.y = point_world.point.y
        pose_msg.pose.position.z = point_world.point.z

        quat = tft.quaternion_from_euler(0, 0, np.deg2rad(angle))
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        rospy.loginfo(f"Published cube pose: {pose_msg}")

    
    def transform_into_world(self, cube_point_camera):
        try:
            now = rospy.Time(0)
            # "left_camera_link_optical"
            # self.tf_listener.waitForTransform("world", "zed2_left_camera_frame", now, rospy.Duration(4.0))
            transformation = self.tf_buffer.lookup_transform("world", "panda_hand", now)
            # translation = transformation.transform.translation
            # rotation = transformation.transform.rotation
            # rospy.loginfo(f"Transformation: {transformation}")
            point_world = tf2_geometry_msgs.do_transform_point(cube_point_camera, transformation)
            return point_world
        except Exception as e:
            rospy.logerr(f"Transformation failed {e}")
            return 


def main():
    try:
        cube_detector = CubeDetector()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
