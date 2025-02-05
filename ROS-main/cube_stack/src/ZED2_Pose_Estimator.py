#From rostopic list, Access the following ZED2 ROS topics:
#-RGB Image: /zed2/zed_node/left/image_rect_color
#-Depth Map: /zed2/zed_node/depth/depth_registered
#-Camera Info: /zed2/zed_node/depth/camera_info

#Detect cubes using color thresholding.
#Calculate their 3D positions using the depth map and camera intrinsics.
#Output the poses in a global frame (world frame) using the /tf transform tree if needed.

#Code implementation below:

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import geometry_msgs.msg

# Initialize OpenCV Bridge
bridge = CvBridge()

# Camera intrinsics
fx, fy, cx, cy = None, None, None, None

def camera_info_callback(msg):
    """Callback to get camera intrinsic parameters from CameraInfo message."""
    global fx, fy, cx, cy
    fx = msg.K[0]  # Focal length x, fx (focal length in pixels, x-axis): 527.2972398956961 (to be used if we cant access K matrix)
    fy = msg.K[4]  # Focal length y, fy (focal length in pixels, y-axis): 527.2972398956961
    cx = msg.K[2]  # Principal point x, cx (principal point x-coordinate): 659.3049926757812
    cy = msg.K[5]  # Principal point y, cy (principal point y-coordinate): 371.39849853515625
    rospy.loginfo("Camera intrinsics received.")
    

def validate_cube_size(x, y, w, h, depth_image):
    """
    Validate if the detected bounding box corresponds to the expected cube size of 4.5cm.
    """
    global fx, fy

    # Get the depth at the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    depth = depth_image[center_y, center_x]

    if depth == 0:  # Skip invalid depth values
        return False

    # Calculate real-world dimensions
    real_width = (w * depth) / fx  # Convert pixel width to real-world width
    real_height = (h * depth) / fy  # Convert pixel height to real-world height

    # Check if dimensions are close to 4.5 cm (tolerance of ±0.5 cm)
    if 4.0 <= real_width <= 5.0 and 4.0 <= real_height <= 5.0:
        return True
    else:
        return False

def detect_cubes(rgb_image, depth_image): ##currently i use a very basic color tracking method to detect cubes, we should replace it with YOLO or ...
    """
    Detect cubes in the RGB image using simple color thresholding.
    Validate detected cubes based on real-world size.
    """
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define the color range for cube detection (light brown/beige tones)
    lower_bound = np.array([10, 50, 50])
    upper_bound = np.array([40, 200, 255])

    # Threshold the image to get a binary mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cube_centers = []
    for contour in contours:
        # Get bounding box for each detected contour
        x, y, w, h = cv2.boundingRect(contour)

        # Validate the bounding box size
        if validate_cube_size(x, y, w, h, depth_image):
            cube_centers.append((x + w // 2, y + h // 2))  # Store center of the bounding box

            # Draw the bounding box on the image (for visualization)
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cube_centers, rgb_image

def get_3d_positions(centers, depth_image):
    """
    Back-project 2D pixel centers to 3D points using the depth map and camera intrinsics.
    Returns a list of 3D positions in the camera frame.
    """
    global fx, fy, cx, cy
    positions = []

    for (u, v) in centers:
        # Get the depth value at the pixel
        depth = depth_image[v, u]  # Note: (u, v) -> (col, row) in images
        if depth == 0:  # Ignore invalid depth values
            rospy.logwarn(f"Invalid depth at pixel ({u}, {v}), skipping.")
            continue

        # Back-project to 3D
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        positions.append((X, Y, Z))

    return positions

def transform_to_world(positions, camera_frame="zed2_camera_frame"):
    """
    Transform 3D positions from the camera frame to the world frame using tf2.
    Returns a list of poses in the world frame.
    """
    world_positions = []
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    try:
        # Get the transformation from the camera frame to the world frame
        transform = tf_buffer.lookup_transform("world", camera_frame, rospy.Time(0), rospy.Duration(1.0))
        for pos in positions:
            # Transform each point
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = camera_frame
            point.point.x, point.point.y, point.point.z = pos

            transformed_point = tf_buffer.transform(point, "world")
            world_positions.append((transformed_point.point.x, transformed_point.point.y, transformed_point.point.z))

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Failed to lookup transform from camera frame to world frame.")

    return world_positions

def image_callback(msg):
    """
    Main callback to process the RGB image and calculate cube poses.
    """
    global fx, fy, cx, cy

    # Ensure camera intrinsics are available
    if fx is None or fy is None:
        rospy.logwarn("Camera intrinsics not available yet!")
        return ## we can skip this code part if we just manually enter the camera parameters in the beginning

    # Convert ROS Image message to OpenCV format
    rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Detect cubes in the image
    cube_centers, annotated_image = detect_cubes(rgb_image)

    # Retrieve depth map
    try:
        depth_msg = rospy.wait_for_message('/zed2/zed_node/depth/depth_registered', Image, timeout=1.0)
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Get 3D positions of the cubes in the camera frame
        positions = get_3d_positions(cube_centers, depth_image)

        # Optionally transform to world frame
        world_positions = transform_to_world(positions)

        # Publish cube poses
        for pos in world_positions:
            cube_msg = Cube()
            cube_msg.position = geometry_msgs.msg.Point(pos[0], pos[1], pos[2])
            cube_msg.orientation = geometry_msgs.msg.Quaternion(0, 0, 0, 1)  # Default orientation
            cube_pub.publish(cube_msg)

        # Display the annotated RGB image
        cv2.imshow("Cube Detection", annotated_image)
        cv2.waitKey(1)

    except rospy.ROSException as e:
        rospy.logwarn("Failed to get depth image!")
        return

    # Display the annotated RGB image
    cv2.imshow("Cube Detection", annotated_image)
    cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("zed2_cube_detector")

    # Subscribe to camera topics
    rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, image_callback)
    rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, camera_info_callback)

    rospy.spin()


## Extra info for my research: 

### When i run "rostopic echo /zed2/zed_node/right_raw/camera_info", I get :

"""

header: 
  seq: 0
  stamp: 
    secs: 1963
    nsecs: 650000000
  frame_id: "right_camera_link_optical"
height: 720  #<image_height>
width: 1280  #<image_width>
distortion_model: "plumb_bob"
D: [-0.040993299, 0.00959359, -0.004429849, 0.000192024, -0.00032088]
K: [527.2972398956961, 0.0, 659.3049926757812, 0.0, 527.2972398956961, 371.39849853515625, 0.0, 0.0, 1.0]   ## 3x3 intrinsic matrix
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]   ## Extrinsic rotation matrix
P: [527.2972398956961, 0.0, 659.3049926757812, -63.275668787483525, 0.0, 527.2972398956961, 371.39849853515625, 0.0, 0.0, 0.0, 1.0, 0.0]   # 3x4 projection matrix
binning_x: 0  
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---

***K (Intrinsic Matrix): #General form

[ fx   0  cx ]      fx, fy: Focal lengths in pixels.
[  0  fy  cy ]      cx, cy: Principal point coordinates (optical center) in the image.
[  0   0   1 ]

#Our ZED2 camera parameters:

K = [ 527.2972398956961    0.0                659.3049926757812  ]
    [   0.0               527.2972398956961   371.39849853515625 ]
    [   0.0                 0.0                1.0              ]

fx (focal length in pixels, x-axis): 527.2972398956961
fy (focal length in pixels, y-axis): 527.2972398956961
cx (principal point x-coordinate): 659.3049926757812
cy (principal point y-coordinate): 371.39849853515625



### When i run "rostopic echo /zed2/zed_node/left/image_rect_color", I get : large human-unreadable data ,arrays of numbers

### When i run "rostopic echo /zed2/zed_node/depth/depth_registered ", I get :large human-unreadable data ,arrays of numbers


"""

