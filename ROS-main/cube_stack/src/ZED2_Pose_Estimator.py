#!/usr/bin/env python3

#From rostopic list, Access the following ZED2 ROS topics:
#-RGB Image: /zed2/zed_node/left/image_rect_color
#-Depth Map: /zed2/zed_node/depth/depth_registered
#-Camera Info: /zed2/zed_node/depth/camera_info


#Code implementation below using rgb and depth msg:
----------------------


import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import tensorflow as tf

class CubeDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('cube_detector', anonymous=True)
        
        # ROS publishers for cube positions and orientations
        self.cube_poses_pub = rospy.Publisher('/cube_poses', Pose, queue_size=10)
        self.cube_digits_pub = rospy.Publisher('/cube_digits', String, queue_size=10)
        
        # Subscribe to ZED2 camera topics (confirm if the msg is correctly written)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/zed2/rgb/image_rect_color', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/zed2/depth/depth_registered', Image, self.depth_callback)
        
        # Store the latest depth image
        self.current_depth = None
        
        # Load MNIST model for digit recognition
        try:
            self.digit_model = tf.keras.models.load_model('/path/to/digit_recognition_model')
            rospy.loginfo("Digit recognition model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load digit model: {e}")
            self.digit_model = None
            
        rospy.loginfo("Cube detector initialized")
        
    def depth_callback(self, depth_msg):
        """Store the latest depth image"""
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Failed to convert depth image: {e}")
    
    def image_callback(self, img_msg):
        """Process RGB image to detect cubes and digits"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            # Detect cubes
            cube_positions = self.detect_cubes(cv_image)
            
            # Process each detected cube
            for i, (x, y, w, h) in enumerate(cube_positions):
                # Extract cube region
                cube_roi = cv_image[y:y+h, x:x+w]
                
                # Get depth at cube center if depth data is available
                z = None
                if self.current_depth is not None:
                    center_y, center_x = y + h//2, x + w//2
                    if 0 <= center_y < self.current_depth.shape[0] and 0 <= center_x < self.current_depth.shape[1]:
                        z = self.current_depth[center_y, center_x]
                
                # Recognize digit on top face
                digit = self.recognize_digit(cube_roi)
                
                # Create and publish cube pose
                if z is not None:
                    pose = Pose()
                    pose.position = Point(x=(x + w/2), y=(y + h/2), z=z)
                    pose.orientation = Quaternion(x=0, y=0, z=0, w=1)  # Default orientation for now
                    self.cube_poses_pub.publish(pose)
                    
                # Publish recognized digit
                self.cube_digits_pub.publish(String(data=str(digit)))
                
                # Visualize detection
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(cv_image, f"Digit: {digit}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the processed image
            cv2.imshow("Cube Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_cubes(self, image):
        """Detect cubes in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape to identify cubes
        cube_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:  # Adjust based on your cube size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # Square-ish shapes (aspect ratio close to 1)
                if 0.7 < aspect_ratio < 1.3:
                    cube_positions.append((x, y, w, h))
        
        return cube_positions
    
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



