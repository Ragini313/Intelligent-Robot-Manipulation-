#!/usr/bin/env python3


import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import tf2_ros
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import message_filters
import tf.transformations as tf_trans

class CubeDetector:
    def __init__(self):
        rospy.init_node('cube_detector', anonymous=True)
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Create synchronizer for RGB and depth
        self.rgb_sub = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        self.point_cloud_sub = message_filters.Subscriber('/zed2/zed_node/point_cloud/cloud_registered', PointCloud2)
        
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.point_cloud_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)
        
        # Publisher for visualization
        self.pose_pub = rospy.Publisher('/cube_poses', geometry_msgs.msg.PoseArray, queue_size=10)
        
        # Camera intrinsics - our camera calibration:
        # fx (focal length in pixels, x-axis): 527.2972398956961
        # fy (focal length in pixels, y-axis): 527.2972398956961
        # cx (principal point x-coordinate): 659.3049926757812
        # cy (principal point y-coordinate): 371.39849853515625

        self.fx = 527.2972398956961
        self.fy = 527.2972398956961
        self.cx = 659.3049926757812
        self.cy = 371.39849853515625
        
        rospy.loginfo("Cube detector initialized.")
    
    def callback(self, rgb_msg, depth_msg, pc_msg):
        try:
            # Convert ROS message to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            # Detect cubes in RGB image
            cube_centers, cube_colors = self.detect_cubes(rgb_image)
            
            if not cube_centers:
                rospy.loginfo("No cubes detected.")
                return
            
            # Create pose array message
            pose_array_msg = geometry_msgs.msg.PoseArray()
            pose_array_msg.header = rgb_msg.header
            
            # Get transform from camera to base
            try:
                transform = self.tf_buffer.lookup_transform(
                    'panda_link0',  # Target frame (robot base)
                    rgb_msg.header.frame_id,  # Source frame (camera frame)
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF Error: {e}")
                return
            
            # Process each detected cube
            for center, color in zip(cube_centers, cube_colors):
                x, y = center
                
                # Get 3D position from pointcloud
                cube_pose = self.get_3d_position(pc_msg, int(x), int(y))
                if cube_pose is None:
                    continue
                
                # Transform pose to robot base frame
                cube_pose = self.transform_pose(cube_pose, transform)
                pose_array_msg.poses.append(cube_pose)
                
                # Print cube information
                pos = cube_pose.position
                rospy.loginfo(f"Cube detected: Color={color}, Position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
            
            # Publish poses
            self.pose_pub.publish(pose_array_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_cubes(self, image):
        """
        Detect cubes using color segmentation
        Returns list of centers and colors
        """
        cube_centers = []
        cube_colors = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different colored cubes
        # Adjust these ranges based on your cube colors
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 150, 100], [140, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255])
        }
        
        # For visualization
        vis_image = image.copy()
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Create mask and find contours
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by contour area
                if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cube_centers.append((cx, cy))
                        cube_colors.append(color_name)
                        
                        # Draw on visualization image
                        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(vis_image, (cx, cy), 5, (255, 255, 255), -1)
                        cv2.putText(vis_image, color_name, (cx - 20, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display result
        cv2.imshow("Cube Detection", vis_image)
        cv2.waitKey(1)
        
        return cube_centers, cube_colors
    
    def get_3d_position(self, pc_msg, x, y):
        """
        Extract 3D position from pointcloud at given pixel coordinates
        """
        try:
            # Sample points in a small region around the pixel
            points = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    points_list = list(pc2.read_points(pc_msg, 
                                                      field_names=("x", "y", "z"),
                                                      skip_nans=True, 
                                                      uvs=[(x+i, y+j)]))
                    if points_list:
                        points.append(points_list[0])
            
            if not points:
                return None
            
            # Average the valid points
            x_sum, y_sum, z_sum = 0, 0, 0
            count = 0
            for point in points:
                x_sum += point[0]
                y_sum += point[1]
                z_sum += point[2]
                count += 1
            
            if count == 0:
                return None
            
            # Create pose message
            pose = geometry_msgs.msg.Pose()
            pose.position.x = x_sum / count
            pose.position.y = y_sum / count
            pose.position.z = z_sum / count
            
            # Set orientation (assuming cube is not rotated)
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1
            
            return pose
            
        except Exception as e:
            rospy.logerr(f"Error extracting 3D position: {e}")
            return None
    
    def transform_pose(self, pose, transform):
        """
        Transform pose from camera frame to robot base frame
        """
        # Extract translation and rotation from transform
        trans = transform.transform.translation
        rot = transform.transform.rotation
        
        # Convert pose to transformation matrix
        pos = pose.position
        quat = pose.orientation
        
        # Apply transform
        transformed_pose = geometry_msgs.msg.Pose()
        
        # Apply rotation
        quat_in = [quat.x, quat.y, quat.z, quat.w]
        quat_tf = [rot.x, rot.y, rot.z, rot.w]
        quat_result = tf_trans.quaternion_multiply(quat_tf, quat_in)
        
        # Apply translation
        transformed_pose.position.x = pos.x + trans.x
        transformed_pose.position.y = pos.y + trans.y
        transformed_pose.position.z = pos.z + trans.z
        
        # Set orientation
        transformed_pose.orientation.x = quat_result[0]
        transformed_pose.orientation.y = quat_result[1]
        transformed_pose.orientation.z = quat_result[2]
        transformed_pose.orientation.w = quat_result[3]
        
        return transformed_pose

if __name__ == '__main__':
    try:
        detector = CubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

