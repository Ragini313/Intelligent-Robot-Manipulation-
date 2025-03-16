#!/usr/bin/env python3

import cv2
import numpy as np
import os
import rospy

class CubeDetector:
    def __init__(self):
        """
        Initialize the cube detector
        """
        pass
    
    def preprocess_image(self, rgb_image):
        """
        Preprocess the image for cube detection
        
        Args:
            rgb_image (np.ndarray): Input RGB image
        
        Returns:
            np.ndarray: Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken edge lines
        dilated = cv2.dilate(edges, None, iterations=2)
        
        return dilated
    
    def detect_cubes(self, rgb_image, depth_image=None):
        """
        Detect cubes in the image
        
        Args:
            rgb_image (np.ndarray): RGB image
            depth_image (np.ndarray, optional): Depth image
        
        Returns:
            list: Detected cube information
        """
        # Preprocess image
        processed = self.preprocess_image(rgb_image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store detected cube information
        detected_cubes = []
        
        for contour in contours:
            # Filter contours by area
            area = cv2.contourArea(contour)
            if area < 500 or area > 1000:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio to ensure cube-like shape
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            # Extract cube ROI
            cube_roi = rgb_image[y:y+h, x:x+w]
            
            # Estimate orientation using minimum area rectangle
            rect = cv2.minAreaRect(contour)
            (center, (width, height), angle) = rect
            
            # Get depth information
            depth_z = None
            if depth_image is not None:
                try:
                    # Use center point for depth
                    center_x, center_y = map(int, center)
                    if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                        depth_z = depth_image[center_y, center_x]
                except Exception:
                    depth_z = None
            
            # Attempt to recognize digits using simple thresholding
            digit = self.recognize_digit(cube_roi)
            
            cube_info = {
                'position': {
                    'x': center[0],
                    'y': center[1],
                    'z': depth_z
                },
                'orientation': angle,  # in degrees
                'digit': digit,
                'bbox': (x, y, w, h)
            }
            
            detected_cubes.append(cube_info)
        
        return detected_cubes
        
    ## This function will be updated with the digit detector neural network ,the code below is a placeholder.
    def recognize_digit(self, cube_image):
        """
        Simple digit recognition using thresholding and contour analysis
        
        Args:
            cube_image (np.ndarray): Image of the cube
        
        Returns:
            str: Recognized digit or '?' if recognition fails
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cube_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely the digit)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Rough digit estimation based on aspect ratio and area
                aspect_ratio = float(w) / h
                
                # These are very rough heuristics and would need calibration
                if 0.3 < aspect_ratio < 0.7:
                    # Likely 1
                    return "1"
                elif 0.7 < aspect_ratio < 1.3:
                    # Likely 0, 4, 7
                    return "0"
                else:
                    # Other digits might have different characteristics
                    return "?"
            
            return "?"
        
        except Exception as e:
            rospy.logerr(f"Digit recognition error: {e}")
            return "?"
    
    def visualize_results(self, rgb_image, detected_cubes):
        """
        Create a visualization of detected cubes
        
        Args:
            rgb_image (np.ndarray): Original RGB image
            detected_cubes (list): List of detected cube information
        
        Returns:
            np.ndarray: Annotated image
        """
        display_image = rgb_image.copy()
        
        for cube in detected_cubes:
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw centroid
            center_x = int(cube['position']['x'])
            center_y = int(cube['position']['y'])
            cv2.circle(display_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Annotate with digit and orientation
            label = f"Digit: {cube['digit']}, Angle: {cube['orientation']:.2f}°"
            cv2.putText(display_image, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display_image

def main():
    # Initialize ROS node
    rospy.init_node('cube_detection_node', anonymous=True)
    
    # Get parameters
    # rgb_path = rospy.get_param('/home/ragini/Desktop/iRobMan Lab/zed2_rgb_raw_image.png', '')
    # depth_path = rospy.get_param('/home/ragini/Desktop/iRobMan Lab/zed2_depth_image.png', '')
    # output_path = rospy.get_param('~output_path', 'cube_detection_result.png')
    rgb_path = '/opt/ros_ws/src/Intelligent-Robot-Manipulation-/rgb_raw_image.png'
    depth_path = '/opt/ros_ws/src/Intelligent-Robot-Manipulation-/image.png'
    output_path = 'cube_detection_result.png'  # or full path if needed
    
    # Validate input paths
    if not rgb_path or not os.path.exists(rgb_path):
        rospy.logerr(f"Invalid RGB image path: {rgb_path}")
        return
    
    if not depth_path or not os.path.exists(depth_path):
        rospy.logerr(f"Invalid depth image path: {depth_path}")
        return
    
    # Load images
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    
    # Initialize detector
    detector = CubeDetector()
    
    # Detect cubes
    detected_cubes = detector.detect_cubes(rgb_image, depth_image)
    
    # Log detection results
    rospy.loginfo(f"Detected {len(detected_cubes)} cubes")
    for i, cube in enumerate(detected_cubes):
        rospy.loginfo(f"Cube {i+1}:")
        rospy.loginfo(f"  Position: {cube['position']}")
        rospy.loginfo(f"  Orientation: {cube['orientation']:.2f}°")
        rospy.loginfo(f"  Digit: {cube['digit']}")
    
    # Visualize results
    result_image = detector.visualize_results(rgb_image, detected_cubes)
    #result_image = detector.visualize_results(rgb_image_cropped, detected_cubes)
    
    # Save result
    cv2.imwrite(output_path, result_image)
    rospy.loginfo(f"Result saved to {output_path}")

if __name__ == '__main__':
    main()


# from PIL import Image

# def crop(image_path, coords, saved_location):
#     image_obj = Image.open("/opt/ros_ws/src/Intelligent-Robot-Manipulation-/rgb_raw_image.png")
#     cropped_image = image_obj.crop(coords)
#     cropped_image.save(saved_location)
#     cropped_image.show()


# if __name__ == '__main__':
#     image = "image.jpg"
#     crop(image, (150, 95, 550, 400 ), 'cropped.jpg')

#     ## 150 - shortens view from left side
#     ## 95 - lowers view from top
#     ## 550 - allows to see the right edge of the table
#     ## the last value just shifts the image vertically
#     ## still have not managed to cut the lower half of the image







