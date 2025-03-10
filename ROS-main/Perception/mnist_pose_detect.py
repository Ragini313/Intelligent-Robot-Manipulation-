#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
import message_filters
from scipy.spatial.transform import Rotation as R

class CubeDetector:
    def __init__(self):
        rospy.init_node('cube_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Load digit recognition model (using pre-trained MNIST model)
        self.digit_model = cv2.ml.KNearest_create()
        self.load_digit_model()
        
        # Publishers
        self.cube_poses_pub = rospy.Publisher('/cube_poses', PoseArray, queue_size=10)
        self.cube_digits_pub = rospy.Publisher('/cube_digits', Int32MultiArray, queue_size=10)
        
        # Synchronize RGB and depth images
        self.rgb_sub = message_filters.Subscriber('/zed2/zed_node/left/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        self.camera_info_sub = rospy.Subscriber('/zed2/zed_node/left/camera_info', CameraInfo, self.camera_info_callback)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)
        
        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        rospy.loginfo("Cube detector initialized. Waiting for images...")
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera calibration parameters received")
    
    def load_digit_model(self):
        
        # Load MNIST dataset or use a pre-trained model
        # For simplicity, let's assume we have a pre-trained model
        # In practice, you would load an actual trained model
        rospy.loginfo("Digit recognition model loaded")
    
    def detect_cubes(self, rgb_img, depth_img):
        # Convert to grayscale for processing
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            rospy.loginfo(f"Found contour with area: {area}")

            # Try with a wider range first to see what's being detected
            # Relaxed thresholds
            if area > 500 and area < 15000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                rospy.loginfo(f"Approximated contour has {len(approx)} vertices")
                if len(approx) >= 4 and len(approx) <= 6:    # More forgiving polygon detection
                    cube_contours.append(contour)   
        
        return cube_contours
    
    def recognize_digit(self, roi):
        # Preprocess the digit image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Resize to match the model input size (e.g., 28x28 for MNIST)
        resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize and reshape
        normalized = resized.astype(np.float32) / 255.0
        sample = normalized.reshape(1, 784)
        
        # Perform prediction
        # For demonstration, we'll return a random digit (0-9)
        # Replace this with our actual model prediction
        # ret, result, neighbours, dist = self.digit_model.findNearest(sample, k=5)
        # digit = int(result[0][0])
        
        # For demo only:
        digit = np.random.randint(0, 10)
        
        return digit
    
    def estimate_pose(self, contour, depth_img):
        # Find the center of the contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        
        # Get center coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get depth at the center
        depth = depth_img[cy, cx] / 1000.0  # Convert to meters if in mm
        rospy.loginfo(f"Depth at center ({cx}, {cy}): {depth_img[cy, cx]}")
        
        if depth <= 0 or np.isnan(depth):
            # Try to find a valid depth nearby
            kernel_size = 5
            x_start = max(0, cx - kernel_size // 2)
            y_start = max(0, cy - kernel_size // 2)
            x_end = min(depth_img.shape[1], cx + kernel_size // 2 + 1)
            y_end = min(depth_img.shape[0], cy + kernel_size // 2 + 1)
            
            depth_window = depth_img[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[depth_window > 0]
            
            if len(valid_depths) > 0:
                depth = np.median(valid_depths) / 1000.0  # Convert to meters
            else:
                return None
        
        # Project pixel to 3D using camera intrinsics
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx_cam = self.camera_matrix[0, 2]
            cy_cam = self.camera_matrix[1, 2]
            
            # Calculate 3D point in camera frame
            x = (cx - cx_cam) * depth / fx
            y = (cy - cy_cam) * depth / fy
            z = depth
            
            # Create pose
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            
            # For orientation, we need to estimate the cube's orientation
            # We can use PnP with the contour corners
            # For simplicity, we'll use an approximate method
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Convert angle to quaternion (assuming rotation around z-axis)
            r = R.from_euler('z', angle, degrees=True)
            quat = r.as_quat()
            
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            return pose
        
        return None
    
    def transform_to_base_frame(self, pose, frame_id="zed2_camera_center"):
        # Transform the pose from camera frame to robot base frame
        try:
            # Create a stamped pose
            pose_stamped = tf2_geometry_msgs.PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = rospy.Time.now()
            
            # Transform to base frame
            transform = self.tf_buffer.lookup_transform(
                "base_link", frame_id, rospy.Time(0), rospy.Duration(1.0))
            
            pose_transformed = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform)
            
            return pose_transformed.pose
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")
            return None
    
    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            rospy.logwarn("Camera calibration parameters not received yet")
            return
        
        try:
            # Convert ROS images to OpenCV format
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            # Detect cubes in the RGB image
            cube_contours = self.detect_cubes(rgb_img, depth_img)
            
            # Prepare messages
            pose_array = PoseArray()
            pose_array.header.frame_id = "base_link"
            pose_array.header.stamp = rospy.Time.now()
            
            digits = Int32MultiArray()
            digits_list = []
            
            # Process each detected cube
            for contour in cube_contours:
                # Draw contour for visualization
                cv2.drawContours(rgb_img, [contour], -1, (0, 255, 0), 2)
                
                # Get bounding rectangle for digit recognition
                x, y, w, h = cv2.boundingRect(contour)
                roi = rgb_img[y:y+h, x:x+w]
                
                # Recognize digit
                digit = self.recognize_digit(roi)
                digits_list.append(digit)
                
                # Draw digit on image
                cv2.putText(rgb_img, str(digit), (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Estimate cube pose in camera frame
                pose = self.estimate_pose(contour, depth_img)
                if pose is not None:
                    # Transform to base frame
                    base_pose = self.transform_to_base_frame(pose)
                    if base_pose is not None:
                        pose_array.poses.append(base_pose)
            
            # Publish results
            if len(pose_array.poses) > 0:
                self.cube_poses_pub.publish(pose_array)
                
                digits.data = digits_list
                self.cube_digits_pub.publish(digits)
                
                rospy.loginfo(f"Published {len(pose_array.poses)} cube poses and digits")
            
            # Display the result (optional)
            cv2.imshow("Cube Detection", rgb_img)
            cv2.waitKey(1)
            # Display the binary and processed images
            cv2.imshow("Binary Image", opening)
            # cv2.imshow("RGB Image with Detections", rgb_img)
            # cv2.waitKey(1)
        
        except Exception as e:
            rospy.logerr(f"Error processing images: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = CubeDetector()
        detector.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass




## ------------------------


# import rospy
# import cv2
# import numpy as np
# import tf
# import tf2_ros
# import tf2_geometry_msgs
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Int32MultiArray
# import message_filters
# from scipy.spatial.transform import Rotation as R

# class CubeDetector:
#     def __init__(self):
#         rospy.init_node('cube_detector', anonymous=True)
        
#         # Initialize CV bridge
#         self.bridge = CvBridge()
        
#         # Initialize TF listener
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
#         # Load digit recognition model (using pre-trained MNIST model)
#         self.digit_model = cv2.ml.KNearest_create()
#         self.load_digit_model()
        
#         # Publishers
#         self.cube_poses_pub = rospy.Publisher('/cube_poses', PoseArray, queue_size=10)
#         self.cube_digits_pub = rospy.Publisher('/cube_digits', Int32MultiArray, queue_size=10)
#         # Add a publisher for the visualization image
#         self.debug_image_pub = rospy.Publisher('/cube_detection/debug_image', Image, queue_size=10)
        
#         # Synchronize RGB and depth images
#         self.rgb_sub = message_filters.Subscriber('/zed2/zed_node/left/image_rect_color', Image)
#         self.depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
#         self.camera_info_sub = rospy.Subscriber('/zed2/zed_node/left/camera_info', CameraInfo, self.camera_info_callback)
        
#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.rgb_sub, self.depth_sub], 10, 0.5)
#         self.ts.registerCallback(self.image_callback)
        
#         # Camera calibration parameters
#         self.camera_matrix = None
#         self.dist_coeffs = None
        
#         # Visualization flag - set to False to disable GUI windows
#         self.enable_visualization = False
        
#         rospy.loginfo("Cube detector initialized. Waiting for images...")
    
#     def camera_info_callback(self, msg):
#         if self.camera_matrix is None:
#             self.camera_matrix = np.array(msg.K).reshape(3, 3)
#             self.dist_coeffs = np.array(msg.D)
#             rospy.loginfo("Camera calibration parameters received")
    
#     def load_digit_model(self):
#         # For demonstration, we're using a simple approach
#         # In a real implementation, you might use a more sophisticated model like a CNN
        
#         # Load MNIST dataset or use a pre-trained model
#         # For simplicity, let's assume we have a pre-trained model
#         # In practice, you would load an actual trained model
#         rospy.loginfo("Digit recognition model loaded")
    
#     def detect_cubes(self, rgb_img, depth_img):
#         # Convert to grayscale for processing
#         gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
#         # Apply adaptive thresholding to get binary image
#         binary = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Apply morphological operations to clean up the image
#         kernel = np.ones((5, 5), np.uint8)
#         opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
#         # Find contours in the binary image
#         contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         cube_contours = []
#         for contour in contours:
#             # Filter contours based on area
#             area = cv2.contourArea(contour)
#             if area > 1000 and area < 10000:  # Adjust based on your setup
#                 # Check if the contour is approximately square
#                 peri = cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
#                 if len(approx) == 4:
#                     cube_contours.append(contour)
        
#         return cube_contours
    
#     def recognize_digit(self, roi):
#         # Preprocess the digit image
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
#         # Resize to match the model input size (e.g., 28x28 for MNIST)
#         resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        
#         # Normalize and reshape
#         normalized = resized.astype(np.float32) / 255.0
#         sample = normalized.reshape(1, 784)
        
#         # Perform prediction
#         # In a real implementation, use your actual model here
#         # For demonstration, we'll return a random digit (0-9)
#         # Replace this with your actual model prediction
#         # ret, result, neighbours, dist = self.digit_model.findNearest(sample, k=5)
#         # digit = int(result[0][0])
        
#         # For demo only:
#         digit = np.random.randint(0, 10)
        
#         return digit
    
#     def estimate_pose(self, contour, depth_img):
#         # Find the center of the contour
#         M = cv2.moments(contour)
#         if M["m00"] == 0:
#             return None
        
#         # Get center coordinates
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
        
#         # Get depth at the center
#         depth = depth_img[cy, cx] / 1000.0  # Convert to meters if in mm
        
#         if depth <= 0 or np.isnan(depth):
#             # Try to find a valid depth nearby
#             kernel_size = 5
#             x_start = max(0, cx - kernel_size // 2)
#             y_start = max(0, cy - kernel_size // 2)
#             x_end = min(depth_img.shape[1], cx + kernel_size // 2 + 1)
#             y_end = min(depth_img.shape[0], cy + kernel_size // 2 + 1)
            
#             depth_window = depth_img[y_start:y_end, x_start:x_end]
#             valid_depths = depth_window[depth_window > 0]
            
#             if len(valid_depths) > 0:
#                 depth = np.median(valid_depths) / 1000.0  # Convert to meters
#             else:
#                 return None
        
#         # Project pixel to 3D using camera intrinsics
#         if self.camera_matrix is not None:
#             fx = self.camera_matrix[0, 0]
#             fy = self.camera_matrix[1, 1]
#             cx_cam = self.camera_matrix[0, 2]
#             cy_cam = self.camera_matrix[1, 2]
            
#             # Calculate 3D point in camera frame
#             x = (cx - cx_cam) * depth / fx
#             y = (cy - cy_cam) * depth / fy
#             z = depth
            
#             # Create pose
#             pose = Pose()
#             pose.position.x = x
#             pose.position.y = y
#             pose.position.z = z
            
#             # For orientation, we need to estimate the cube's orientation
#             # We can use PnP with the contour corners
#             # For simplicity, we'll use an approximate method
#             rect = cv2.minAreaRect(contour)
#             angle = rect[2]
            
#             # Convert angle to quaternion (assuming rotation around z-axis)
#             r = R.from_euler('z', angle, degrees=True)
#             quat = r.as_quat()
            
#             pose.orientation.x = quat[0]
#             pose.orientation.y = quat[1]
#             pose.orientation.z = quat[2]
#             pose.orientation.w = quat[3]
            
#             return pose
        
#         return None
    
#     def transform_to_base_frame(self, pose, frame_id="zed2_camera_center"):
#         # Transform the pose from camera frame to robot base frame
#         try:
#             # Create a stamped pose
#             pose_stamped = tf2_geometry_msgs.PoseStamped()
#             pose_stamped.pose = pose
#             pose_stamped.header.frame_id = frame_id
#             pose_stamped.header.stamp = rospy.Time.now()
            
#             # Transform to base frame
#             transform = self.tf_buffer.lookup_transform(
#                 "base_link", frame_id, rospy.Time(0), rospy.Duration(1.0))
            
#             pose_transformed = tf2_geometry_msgs.do_transform_pose(
#                 pose_stamped, transform)
            
#             return pose_transformed.pose
        
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
#                 tf2_ros.ExtrapolationException) as e:
#             rospy.logwarn(f"Transform error: {e}")
#             return None
    
#     def image_callback(self, rgb_msg, depth_msg):
#         if self.camera_matrix is None:
#             rospy.logwarn("Camera calibration parameters not received yet")
#             return
        
#         try:
#             # Convert ROS images to OpenCV format
#             rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
#             depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
#             # Detect cubes in the RGB image
#             cube_contours = self.detect_cubes(rgb_img, depth_img)
            
#             # Create a copy for visualization
#             vis_img = rgb_img.copy()
            
#             # Prepare messages
#             pose_array = PoseArray()
#             pose_array.header.frame_id = "base_link"
#             pose_array.header.stamp = rospy.Time.now()
            
#             digits = Int32MultiArray()
#             digits_list = []
            
#             # Process each detected cube
#             for contour in cube_contours:
#                 # Draw contour for visualization
#                 cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
                
#                 # Get bounding rectangle for digit recognition
#                 x, y, w, h = cv2.boundingRect(contour)
#                 roi = rgb_img[y:y+h, x:x+w]
                
#                 # Recognize digit
#                 digit = self.recognize_digit(roi)
#                 digits_list.append(digit)
                
#                 # Draw digit on image
#                 cv2.putText(vis_img, str(digit), (x, y-10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
#                 # Estimate cube pose in camera frame
#                 pose = self.estimate_pose(contour, depth_img)
#                 if pose is not None:
#                     # Transform to base frame
#                     base_pose = self.transform_to_base_frame(pose)
#                     if base_pose is not None:
#                         pose_array.poses.append(base_pose)
                        
#                         # Draw 3D coordinates on the image
#                         cv2.putText(vis_img, 
#                                     f"({base_pose.position.x:.2f}, {base_pose.position.y:.2f}, {base_pose.position.z:.2f})",
#                                     (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
#             # Publish results
#             if len(pose_array.poses) > 0:
#                 self.cube_poses_pub.publish(pose_array)
                
#                 digits.data = digits_list
#                 self.cube_digits_pub.publish(digits)
                
#                 rospy.loginfo(f"Published {len(pose_array.poses)} cube poses and digits")
            
#             # Publish the visualization image instead of showing it
#             try:
#                 debug_img_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
#                 self.debug_image_pub.publish(debug_img_msg)
#             except Exception as e:
#                 rospy.logerr(f"Error publishing visualization: {e}")
            
#             # Only show visualization if enabled (disabled by default)
#             if self.enable_visualization:
#                 try:
#                     cv2.imshow("Cube Detection", vis_img)
#                     cv2.waitKey(1)
#                 except Exception as e:
#                     rospy.logerr(f"Error showing visualization: {e}")
#                     # Disable visualization if it fails
#                     self.enable_visualization = False
        
#         except Exception as e:
#             rospy.logerr(f"Error processing images: {e}")
    
#     def run(self):
#         rospy.spin()

# if __name__ == '__main__':
#     try:
#         detector = CubeDetector()
#         detector.run()
#     except rospy.ROSInterruptException:
#         if detector.enable_visualization:
#             cv2.destroyAllWindows()
#         pass
