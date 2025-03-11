#!/usr/bin/env python3


  # Get parameters:
    # rgb_path = rospy.get_param('/home/ragini/Desktop/iRobMan Lab/zed2_rgb_raw_image.png', '')
    # depth_path = rospy.get_param('/home/ragini/Desktop/iRobMan Lab/zed2_depth_image.png', '')
    # output_path = rospy.get_param('~output_path', 'cube_detection_result.png')
    
import cv2
import numpy as np
import os
import rospy

class CubeDetector:
    def __init__(self, cube_size_cm=4.5):
        """
        Initialize the cube detector
        
        Args:
            cube_size_cm (float): Size of the cube in centimeters
        """
        self.cube_size_cm = cube_size_cm
        
        # ZED2 depth scale factor (to convert depth values to meters)
        # Adjust based on your depth image units
        self.depth_scale = 0.001  # Assuming depth is in millimeters
        
        # Camera intrinsic parameters (replace with calibrated values)
        self.fx = 527.2972398956961  # Focal length in x (pixels)
        self.fy = 527.2972398956961  # Focal length in y (pixels)
        self.cx = 659.3049926757812  # Optical center x (pixels)
        self.cy = 371.39849853515625  # Optical center y (pixels)
    
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
        
        # Debug variable to track rejected contours
        rejected_reason = {
            'area': 0,
            'aspect_ratio': 0
        }
        
        for contour in contours:
            # Filter contours by area
            area = cv2.contourArea(contour)
            if area < 500 or area > 1000:
                rejected_reason['area'] += 1
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio to ensure cube-like shape
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                rejected_reason['aspect_ratio'] += 1
                continue
            
            # Extract cube ROI
            cube_roi = rgb_image[y:y+h, x:x+w]
            
            # Estimate orientation using minimum area rectangle
            rect = cv2.minAreaRect(contour)
            (center, (width, height), angle) = rect
            
            # Get depth information
            depth_z = None
            real_world_pos = None
            
            if depth_image is not None:
                try:
                    # Use center point for depth
                    center_x, center_y = map(int, center)
                    
                    # Ensure center point is within depth image bounds
                    if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                        depth_z = depth_image[center_y, center_x] * self.depth_scale  # Convert to meters
                        
                        # Calculate real-world position using camera intrinsics
                        x_world = (center_x - self.cx) * depth_z / self.fx
                        y_world = (center_y - self.cy) * depth_z / self.fy
                        z_world = depth_z
                        
                        real_world_pos = {
                            'x': x_world,
                            'y': y_world,
                            'z': z_world
                        }
                
                except Exception as e:
                    rospy.logerr(f"Depth processing error: {e}")
                    depth_z = None
            
            # Store cube information
            cube_info = {
                'position': {
                    'x': center[0],
                    'y': center[1],
                    'z': depth_z
                },
                'real_world_position': real_world_pos,
                'orientation': angle,  # in degrees
                'bbox': (x, y, w, h)
            }
            
            detected_cubes.append(cube_info)
        
        # Log rejection reasons for debugging
        rospy.loginfo(f"Rejected contours: {rejected_reason}")
        
        return detected_cubes
    
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
        
        for i, cube in enumerate(detected_cubes):
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw centroid
            center_x = int(cube['position']['x'])
            center_y = int(cube['position']['y'])
            cv2.circle(display_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Annotate with info
            label_pixel = f"Cube {i+1}: ({center_x}, {center_y})"
            cv2.putText(display_image, label_pixel, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add real-world position if available
            if cube['real_world_position']:
                rwp = cube['real_world_position']
                label_world = f"({rwp['x']:.3f}m, {rwp['y']:.3f}m, {rwp['z']:.3f}m)"
                cv2.putText(display_image, label_world, (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add orientation
            angle_label = f"Angle: {cube['orientation']:.1f}°"
            cv2.putText(display_image, angle_label, (x, y+h+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display_image

def main():
    # Initialize ROS node
    rospy.init_node('cube_detection_node', anonymous=True)
    
    # Get parameters
    rgb_path = '/opt/ros_ws/src/Intelligent-Robot-Manipulation-/rgb_raw_image.png'
    depth_path = '/opt/ros_ws/src/Intelligent-Robot-Manipulation-/image.png'
    output_path = 'cube_detection_result.png'
    
    # Add cube size parameter (in cm)
    cube_size = rospy.get_param('~cube_size', 4.5)  # Default to 4.5cm cubes
    
    # Debug flag to save processed image
    save_debug_image = True
    
    # Validate input paths
    if not rgb_path or not os.path.exists(rgb_path):
        rospy.logerr(f"Invalid RGB image path: {rgb_path}")
        return
    
    if depth_path and not os.path.exists(depth_path):
        rospy.logwarn(f"Invalid depth image path: {depth_path}, proceeding without depth")
        depth_image = None
    else:
        # Load depth image
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Load RGB image
    rgb_image = cv2.imread(rgb_path)
    
    if rgb_image is None:
        rospy.logerr("Failed to load RGB image")
        return
    
    # Initialize detector with cube size
    detector = CubeDetector(cube_size_cm=cube_size)
    
    # Save preprocessed image for debugging
    if save_debug_image:
        processed = detector.preprocess_image(rgb_image)
        cv2.imwrite('debug_processed.png', processed)
        rospy.loginfo("Saved preprocessed image to debug_processed.png")
    
    # Detect cubes
    detected_cubes = detector.detect_cubes(rgb_image, depth_image)
    
    # Log detection results
    rospy.loginfo(f"Detected {len(detected_cubes)} cubes")
    for i, cube in enumerate(detected_cubes):
        rospy.loginfo(f"Cube {i+1}:")
        rospy.loginfo(f"  Pixel Position: ({cube['position']['x']:.1f}, {cube['position']['y']:.1f})")
        
        if cube['real_world_position']:
            rwp = cube['real_world_position']
            rospy.loginfo(f"  Real-world Position (m): ({rwp['x']:.3f}, {rwp['y']:.3f}, {rwp['z']:.3f})")
        
        rospy.loginfo(f"  Orientation: {cube['orientation']:.2f}°")
    
    # Visualize results
    result_image = detector.visualize_results(rgb_image, detected_cubes)
    
    # Save result
    cv2.imwrite(output_path, result_image)
    rospy.loginfo(f"Result saved to {output_path}")
    
    # If no cubes detected, try to save a debug image with all contours
    if len(detected_cubes) == 0 and save_debug_image:
        processed = detector.preprocess_image(rgb_image)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_image = rgb_image.copy()
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite('debug_all_contours.png', debug_image)
        rospy.loginfo(f"Saved debug image with all contours to debug_all_contours.png")
        rospy.loginfo(f"Total contours found: {len(contours)}")

if __name__ == '__main__':
    main()



### Output is:
# Saved preprocessed image to debug_processed.png
# [INFO] [1741687914.085557, 231.750000]: Rejected contours: {'area': 21, 'aspect_ratio': 2}
# [INFO] [1741687914.087350, 231.752000]: Detected 7 cubes
# [INFO] [1741687914.089015, 231.754000]: Cube 1:
# [INFO] [1741687914.091175, 231.757000]:   Pixel Position: (476.6, 331.8)
# [INFO] [1741687914.092708, 231.759000]:   Real-world Position (m): (-0.016, -0.003, 0.045)
# [INFO] [1741687914.094166, 231.760000]:   Orientation: -39.47°
# [INFO] [1741687914.095792, 231.762000]: Cube 2:
# [INFO] [1741687914.097353, 231.764000]:   Pixel Position: (315.0, 215.0)
# [INFO] [1741687914.099291, 231.766000]:   Real-world Position (m): (-0.009, -0.004, 0.014)
# [INFO] [1741687914.103102, 231.768000]:   Orientation: -0.00°
# [INFO] [1741687914.104669, 231.770000]: Cube 3:
# [INFO] [1741687914.107513, 231.772000]:   Pixel Position: (360.7, 190.8)
# [INFO] [1741687914.109201, 231.773000]:   Real-world Position (m): (-0.008, -0.005, 0.014)
# [INFO] [1741687914.112143, 231.777000]:   Orientation: -33.69°
# [INFO] [1741687914.113763, 231.779000]: Cube 4:
# [INFO] [1741687914.115690, 231.781000]:   Pixel Position: (269.4, 174.5)
# [INFO] [1741687914.117248, 231.783000]:   Real-world Position (m): (-0.010, -0.005, 0.013)
# [INFO] [1741687914.119163, 231.785000]:   Orientation: -36.87°
# [INFO] [1741687914.120981, 231.787000]: Cube 5:
# [INFO] [1741687914.122617, 231.789000]:   Pixel Position: (317.2, 170.6)
# [INFO] [1741687914.124476, 231.791000]:   Real-world Position (m): (-0.008, -0.005, 0.013)
# [INFO] [1741687914.130927, 231.794000]:   Orientation: -70.71°
# [INFO] [1741687914.140500, 231.801000]: Cube 6:
# [INFO] [1741687914.143334, 231.804000]:   Pixel Position: (608.7, 163.3)
# [INFO] [1741687914.145408, 231.806000]:   Real-world Position (m): (-0.000, -0.000, 0.000)
# [INFO] [1741687914.147021, 231.808000]:   Orientation: -27.41°
# [INFO] [1741687914.148767, 231.810000]: Cube 7:
# [INFO] [1741687914.150520, 231.812000]:   Pixel Position: (378.5, 21.0)
# [INFO] [1741687914.152120, 231.814000]:   Real-world Position (m): (-0.000, -0.000, 0.000)
# [INFO] [1741687914.153937, 231.815000]:   Orientation: -0.00°
# [INFO] [1741687914.208182, 231.842000]: Result saved to cube_detection_result.png
