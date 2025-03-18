#!/usr/bin/env python3


import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters
import tf.transformations as tft

class CubeDetector:
    def __init__(self):
        rospy.init_node('cube_detector', anonymous=True)
        
        # Initialize CV bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Color range for white and yellow cubes in HSV space
        # White: low saturation, high value
        self.white_range = (np.array([0, 0, 180]), np.array([180, 30, 255]))
        # Yellow: specific hue range with moderate saturation and high value
        self.yellow_range = (np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        # Create a TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Camera matrix and distortion coefficients will be populated from camera_info (msg)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # The size of the cube (in meters)
        self.cube_size = 0.05  # real cube is 4.5cm , used 5cm for tolerances
        
        # Minimum contour area to consider (in pixels²)
        self.min_contour_area = 1000  # I increased to filter out small corners
        
        # Publishers for visualization
        self.marker_pub = rospy.Publisher('/cube_markers', MarkerArray, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/cube_detection/debug_image', Image, queue_size=10)
        self.mask_pub = rospy.Publisher('/cube_detection/mask', Image, queue_size=10)
        
        # Subscribe to camera info once to get camera matrix
        rospy.Subscriber('/zed2/zed_node/rgb/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        
        # Wait until we have the camera matrix
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown() and self.camera_matrix is None:
            rospy.loginfo("Waiting for camera info...")
            rate.sleep()
            
        # Synchronized subscribers for RGB and depth images
        self.rgb_sub = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        
        # Time synchronizer with larger queue size and slop
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 30, 0.5)
        self.ts.registerCallback(self.image_callback)
        
        rospy.loginfo("Cube detector initialized!")
        
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera matrix acquired")
    
    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS images to OpenCV format
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            # Create a copy of the RGB image for visualization
            debug_img = cv_rgb.copy()
            
            # Log that we received images
            rospy.loginfo(f"Processing frame with timestamp: {rgb_msg.header.stamp.to_sec()}")
            
            # Detect and compute cube poses
            cube_poses = self.detect_cubes(cv_rgb, cv_depth, debug_img, rgb_msg.header.frame_id)
            
            # Publish visualization markers
            self.publish_cube_markers(cube_poses, rgb_msg.header.frame_id)
            
            # Publish debug image
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            
            # Print cube poses
            if cube_poses:
                rospy.loginfo(f"Detected {len(cube_poses)} cubes")
                for i, pose in enumerate(cube_poses):
                    position = pose.pose.position
                    orientation = pose.pose.orientation
                    rospy.loginfo(f"Cube {i}: Position: [{position.x:.3f}, {position.y:.3f}, {position.z:.3f}], " 
                                f"Orientation: [{orientation.x:.3f}, {orientation.y:.3f}, {orientation.z:.3f}, {orientation.w:.3f}]")
            else:
                rospy.loginfo("No cubes detected")
                
        except Exception as e:
            rospy.logerr(f"Error processing images: {e}")
    
    def detect_cubes(self, rgb_img, depth_img, debug_img, frame_id):
        # Convert RGB image to HSV for better color segmentation
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        
        # Create mask for white cubes
        white_mask = cv2.inRange(hsv_img, self.white_range[0], self.white_range[1])
        
        # Create mask for yellow cubes
        yellow_mask = cv2.inRange(hsv_img, self.yellow_range[0], self.yellow_range[1])
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional morphological closing to connect nearby components
        kernel_close = np.ones((15, 15), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Publish the mask for debugging
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(combined_mask, "mono8"))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        cube_poses = []
        
        # Process top N largest contours
        for i, contour in enumerate(contours[:10]):  # Process up to 10 largest contours
            area = cv2.contourArea(contour)
            
            # Filter out small contours - significantly increased threshold
            if area > self.min_contour_area:
                # Check if the contour is square-like
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # If it has 4 corners, it's likely a square/rectangle
                # Or if it's a more complex shape but has reasonable aspect ratio
                if len(approx) >= 4 and len(approx) <= 8:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Check if it's reasonably square-like (not too elongated)... does this enhance the results? if not, delete
                    if 0.7 <= aspect_ratio <= 1.3:
                        # Draw contour on debug image
                        contour_color = (255, 255, 0)  # BGR: Cyan
                        cv2.drawContours(debug_img, [contour], -1, contour_color, 2)
                        
                        # Find the minimum area rectangle around the contour
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(debug_img, [box], 0, contour_color, 2)
                        
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = 0, 0
                            
                        # Mark the center on the debug image
                        cv2.circle(debug_img, (cx, cy), 5, contour_color, -1)
                        cv2.putText(debug_img, f"Cube {i}", (cx - 20, cy - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 2)
                        
                        # Get depth at the center of the contour
                        # Apply a larger averaging window to reduce noise
                        window_size = 5
                        depth_window = depth_img[max(0, cy-window_size):min(depth_img.shape[0], cy+window_size+1),
                                                max(0, cx-window_size):min(depth_img.shape[1], cx+window_size+1)]
                        
                        # Filter out invalid/NaN depth values
                        valid_depths = depth_window[(~np.isnan(depth_window)) & (depth_window > 0) & (depth_window < 1.0)]
                        
                        if len(valid_depths) > 0:
                            # Get median depth in meters
                            depth_val = np.median(valid_depths)
                            
                            # Filter out cubes that are too far or too close
                            if 0.2 <= depth_val <= 0.8:  # Adjust these thresholds as needed
                                # Display depth on image
                                cv2.putText(debug_img, f"D: {depth_val:.3f}m", (cx - 20, cy + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 2)
                                
                                # Display area on image
                                cv2.putText(debug_img, f"A: {area:.0f}", (cx - 20, cy + 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 2)
                                
                                # Create a pose for the cube
                                cube_pose = self.create_cube_pose(cx, cy, depth_val, rect, frame_id)
                                if cube_pose is not None:
                                    cube_poses.append(cube_pose)
        
        return cube_poses

    # def load_digit_model(self):
    
    # # Load MNIST dataset or use a pre-trained model
    # # For simplicity, let's assume we have a pre-trained model
    # # In practice, you would load an actual trained model
    # rospy.loginfo("Digit recognition model loaded")


    # def recognize_digit(self, roi):
    #     # Preprocess the digit image
    #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    #     # Resize to match the model input size (e.g., 28x28 for MNIST)
    #     resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        
    #     # Normalize and reshape
    #     normalized = resized.astype(np.float32) / 255.0
    #     sample = normalized.reshape(1, 784)
        
    #     # Perform prediction
    #     # For demonstration, we'll return a random digit (0-9)
    #     # Replace this with our actual model prediction
    #     # ret, result, neighbours, dist = self.digit_model.findNearest(sample, k=5)
    #     # digit = int(result[0][0])
        
    #     # For demo only:
    #     digit = np.random.randint(0, 10)
        
    #     return digit
    
    def create_cube_pose(self, cx, cy, depth, rect, frame_id):
        try:
            # Project 2D point to 3D using the camera matrix
            # [X, Y, Z] = depth * inv(camera_matrix) * [cx, cy, 1]
            point_3d = np.array([(cx - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0],
                                (cy - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1],
                                depth])
            
            # Create a pose
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = rospy.Time.now()
            
            # Set position
            pose.pose.position.x = point_3d[0]
            pose.pose.position.y = point_3d[1]
            pose.pose.position.z = point_3d[2]
            
            # Set orientation based on the rectangle angle
            angle = np.deg2rad(rect[2])
            
            # Create a rotation matrix (simple Z rotation based on rect angle)
            # Better pose estimation would use PnP with multiple points
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            # Convert rotation matrix to quaternion
            quaternion = tft.quaternion_from_matrix(rotation_matrix)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            
            return pose
            
        except Exception as e:
            rospy.logerr(f"Error creating cube pose: {e}")
            return None
    
    def publish_cube_markers(self, cube_poses, frame_id):
        marker_array = MarkerArray()
        
        for i, pose in enumerate(cube_poses):
            # Create a cube marker
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "cubes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set the pose from our detection
            marker.pose = pose.pose
            
            # Set scale based on cube size
            marker.scale.x = self.cube_size
            marker.scale.y = self.cube_size
            marker.scale.z = self.cube_size
            
            # Set color to white or yellow based on the detected color
            # You can add logic here to determine the color based on the mask
            # For simplicity, we'll set it to white
            marker.color = ColorRGBA(1.0, 1.0, 1.0, 0.7)  # RGBA: White
                
            marker.lifetime = rospy.Duration(0.5)  # Display for 0.5 seconds
            
            marker_array.markers.append(marker)
            
        # Publish the marker array
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def set_color_range(self, lower_hsv, upper_hsv):
        """Utility method to update color thresholds at runtime"""
        self.color_range = (np.array(lower_hsv), np.array(upper_hsv))
        rospy.loginfo(f"Updated color range to: {lower_hsv} - {upper_hsv}")

def main():
    try:
        detector = CubeDetector()
        # You can tune these parameters at runtime if needed
        # detector.set_color_range([0, 0, 180], [180, 30, 255])  # White in HSV
        # detector.min_contour_area = 1000  # Minimum area to consider
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
