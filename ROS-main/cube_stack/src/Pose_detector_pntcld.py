#!/usr/bin/env python3

#Code implementation below using pointcloud msg:
----------------------

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray
from std_msgs.msg import String, Int32MultiArray
import tensorflow as tf
import tf.transformations
import message_filters

class CubeDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('cube_detector', anonymous=True)
        
        # ROS publishers
        self.cube_poses_pub = rospy.Publisher('/cube_poses', PoseArray, queue_size=10)
        self.cube_digits_pub = rospy.Publisher('/cube_digits', Int32MultiArray, queue_size=10)
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Use message filters to synchronize RGB image and point cloud
        self.image_sub = message_filters.Subscriber('/zed2/rgb/image_rect_color', Image)
        self.pointcloud_sub = message_filters.Subscriber('/zed2/point_cloud/cloud_registered', PointCloud2)
        
        # Create a time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub], 10, 0.1)
        self.ts.registerCallback(self.synchronized_callback)
        
        # Load MNIST model for digit recognition
        try:
            self.digit_model = tf.keras.models.load_model('/path/to/digit_recognition_model')
            rospy.loginfo("Digit recognition model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load digit model: {e}")
            self.digit_model = None
            
        # RANSAC parameters for plane fitting
        self.ransac_iterations = 100
        self.ransac_threshold = 0.01  # 1cm
        
        rospy.loginfo("Cube detector initialized with point cloud processing")
        
    def synchronized_callback(self, img_msg, pointcloud_msg):
        """Process synchronized RGB image and point cloud data"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            # Detect potential cubes in the image
            cube_regions = self.detect_cube_regions(cv_image)
            
            # Extract point cloud data
            pc_data = pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            pc_array = np.array(list(pc_data))
            
            # Create structured point cloud (with image coordinates)
            # This assumes the point cloud and image are aligned pixel-by-pixel
            h, w = cv_image.shape[:2]
            pc_structured = np.zeros((h, w, 3), dtype=np.float32)
            pc_structured.fill(np.nan)
            
            # Restructure point cloud to match image coordinates
            for point in pc_array:
                # Convert 3D point to pixel coordinates
                # Note: This is a simple approach. For more accuracy, use the camera's projection matrix
                u = int((point[0] / point[2]) * w / 2 + w / 2)
                v = int((point[1] / point[2]) * h / 2 + h / 2)
                
                if 0 <= u < w and 0 <= v < h:
                    pc_structured[v, u] = [point[0], point[1], point[2]]
            
            # Process each detected cube
            poses = []
            digits = []
            
            for i, (x, y, w, h) in enumerate(cube_regions):
                # Extract cube region
                cube_roi = cv_image[y:y+h, x:x+w]
                
                # Extract 3D points for this region
                cube_points = pc_structured[y:y+h, x:x+w].reshape(-1, 3)
                cube_points = cube_points[~np.isnan(cube_points).any(axis=1)]  # Remove NaN values
                
                if len(cube_points) < 10:  # Not enough points
                    continue
                
                # Fit a plane to the top surface using RANSAC
                plane_model, inliers = self.ransac_plane_fit(cube_points)
                
                if plane_model is not None:
                    # Get 3D position (centroid of top face)
                    top_face_points = cube_points[inliers]
                    centroid = np.mean(top_face_points, axis=0)
                    
                    # Calculate orientation from plane normal
                    normal = plane_model[:3]
                    # Convert normal to quaternion (align z-axis with normal)
                    rotation = self.normal_to_quaternion(normal)
                    
                    # Create pose
                    pose = Pose()
                    pose.position = Point(x=centroid[0], y=centroid[1], z=centroid[2])
                    pose.orientation = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
                    poses.append(pose)
                    
                    # Recognize digit
                    digit = self.recognize_digit(cube_roi)
                    digits.append(int(digit) if digit.isdigit() else -1)
                    
                    # Visualize detection
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"Digit: {digit}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish results
            if poses:
                pose_array = PoseArray()
                pose_array.header.stamp = rospy.Time.now()
                pose_array.header.frame_id = img_msg.header.frame_id
                pose_array.poses = poses
                self.cube_poses_pub.publish(pose_array)
                
                digit_array = Int32MultiArray()
                digit_array.data = digits
                self.cube_digits_pub.publish(digit_array)
            
            # Display the processed image
            cv2.imshow("Cube Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing synchronized data: {e}")
    
    def detect_cube_regions(self, image):
        """Detect potential cube regions in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape to identify cubes
        cube_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:  # Adjust based on your cube size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # Square-ish shapes (aspect ratio close to 1)
                if 0.7 < aspect_ratio < 1.3:
                    cube_regions.append((x, y, w, h))
        
        return cube_regions
    
    def ransac_plane_fit(self, points, threshold=0.01, iterations=100):
        """
        Fit a plane to 3D points using RANSAC
        Returns: [a, b, c, d] for plane equation ax + by + cz + d = 0, and indices of inliers
        """
        best_inliers = []
        best_model = None
        n_points = points.shape[0]
        
        for _ in range(iterations):
            # Randomly select 3 points to define a plane
            indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[indices]
            
            # Calculate plane equation from 3 points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)  # Normalize
            d = -np.dot(normal, p1)
            plane = np.array([normal[0], normal[1], normal[2], d])
            
            # Count inliers
            distances = np.abs(np.dot(points, plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])
            inliers = np.where(distances < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = plane
                
        # Refine the model using all inliers
        if len(best_inliers) > 3:
            inlier_points = points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            
            # Use SVD for plane fitting
            centered_points = inlier_points - centroid
            _, _, vh = np.linalg.svd(centered_points)
            normal = vh[2, :]
            
            d = -np.dot(normal, centroid)
            refined_model = np.array([normal[0], normal[1], normal[2], d])
            
            return refined_model, best_inliers
        
        return best_model, best_inliers
    
    def normal_to_quaternion(self, normal):
        """Convert a normal vector to a quaternion that aligns the z-axis with the normal"""
        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)
        
        # Find rotation axis between [0, 0, 1] and normal
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, normal)
        
        # If normals are parallel, set a default axis
        if np.linalg.norm(axis) < 1e-6:
            if np.dot(normal, z_axis) > 0:
                return [0, 0, 0, 1]  # Identity quaternion
            else:
                return [1, 0, 0, 0]  # 180° rotation around x-axis
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        # Calculate rotation angle
        angle = np.arccos(np.dot(normal, z_axis))
        
        # Create quaternion from axis-angle
        return tf.transformations.quaternion_about_axis(angle, axis)
    
    def recognize_digit(self, cube_image):
        """Recognize the digit on top of the cube"""
        if self.digit_model is None:
            return "?"
            
        # Process for digit recognition:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(cube_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply thresholding to isolate the digit
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Find contours to locate the digit
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the digit)
            digit_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(digit_contour)
            
            # Extract digit region with some margin
            margin = 4  # pixels
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(cube_image.shape[1], x + w + margin)
            y_end = min(cube_image.shape[0], y + h + margin)
            
            digit_roi = thresh[y_start:y_end, x_start:x_end]
            
            # Resize to match input size expected by model (28x28 for MNIST)
            if digit_roi.size > 0:
                digit_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Normalize and prepare for model input
                digit_roi = digit_roi.astype('float32') / 255.0
                digit_roi = np.expand_dims(digit_roi, axis=0)
                digit_roi = np.expand_dims(digit_roi, axis=-1)
                
                # Predict using model
                prediction = self.digit_model.predict(digit_roi)
                return str(np.argmax(prediction[0]))
        
        return "?"

    def run(self):
        """Run the node"""
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = CubeDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
