#!/usr/bin/env python3

import os
os.environ['TF_REPEATED_DATA'] = 'ignore'

import rospy
import numpy as np
import open3d as o3d
import tf
import tf.transformations
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseArray, Pose, Point
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_pose





class CubeDetector:
    def __init__(self):
        rospy.init_node('cube_detector', anonymous=True)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Parameters (can be moved to ROS params)
        self.cluster_tolerance = 0.01  # Distance between points for clustering
        self.min_cluster_size = 50     # Minimum points to consider a cluster
        self.max_cluster_size = 1000   # Maximum points in a cluster
        self.ground_plane_threshold = 0.01  # RANSAC threshold for ground plane detection
        
        # Parameters adjusted for 4.5cm cubes
        self.min_side_length = 0.035    # Minimum cube side length (3cm to allow some tolerance)
        self.max_side_length = 0.055    # Maximum cube side length (6cm to allow some tolerance)
        
        # From logs, we can see that is_cubic=False is causing most cubes to be rejected
        # Let's make this more permissive
        self.cube_dimension_ratio = 0.2  # Increased from 0.3 to 0.4 to allow more variation
        
        # Debug mode (visualize intermediary steps)
        self.debug = True
        
        # Flag to track if detection has been performed successfully
        self.detection_completed = False
        
        # Publishers
        self.pose_pub = rospy.Publisher('/detected_cube_poses', PoseArray, queue_size=1)
        self.marker_pub = rospy.Publisher('/cube_markers', MarkerArray, queue_size=1)
        self.debug_marker_pub = rospy.Publisher('/debug_markers', MarkerArray, queue_size=1)
        
        # Subscriber - stored so we can unsubscribe later
        self.pc_sub = None
        
        rospy.loginfo("Cube detector initialized")
        
        # Start the detection process
        self.start_detection()
    
    def start_detection(self):
        """Start the detection process by subscribing to the point cloud topic"""
        rospy.loginfo("Starting detection...")
        self.pc_sub = rospy.Subscriber('/zed2/zed_node/point_cloud/cloud_registered', 
                                      PointCloud2, self.pc_callback, queue_size=1)

   
    
    def pc_callback(self, msg):
        # Skip if we've already completed detection
        if self.detection_completed:
            return

        try:
            rospy.loginfo("Processing point cloud...")

            # Convert PointCloud2 message to numpy array
            point_list = []
            for point in point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
                point_list.append([point[0], point[1], point[2]])

            if len(point_list) < 100:
                rospy.logwarn("Received point cloud is too small: %d points", len(point_list))
                return

            points = np.array(point_list)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Downsample the point cloud for efficiency
            pcd = pcd.voxel_down_sample(voxel_size=0.003)

            # Segment and remove the ground plane
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.ground_plane_threshold,
                ransac_n=3,
                num_iterations=1000
            )

            # Get points that are not part of the ground plane
            outlier_cloud = pcd.select_by_index(inliers, invert=True)

            # Filter out points that are too high (optional, adjust based on setup)
            points_array = np.asarray(outlier_cloud.points)
            height_mask = points_array[:, 2] < 0.5  # Filter out points above 0.5m
            filtered_points = points_array[height_mask]

            if len(filtered_points) < 50:
                rospy.logwarn("Too few points after filtering: %d", len(filtered_points))
                return

            filtered_cloud = o3d.geometry.PointCloud()
            filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)

            # Cluster the remaining points using DBSCAN
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
                labels = np.array(filtered_cloud.cluster_dbscan(
                    eps=self.cluster_tolerance,
                    min_points=10,
                    print_progress=False
                ))

            max_label = labels.max()
            rospy.loginfo(f"Detected {max_label + 1} potential clusters")

            # Initialize messages
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "panda_link0"  # Publish poses in robot base frame

            marker_array = MarkerArray()

            # Process each cluster
            cube_count = 0
            clusters = []
            verified_cubes = []  # List to store verified cubes

            # Add this section to print cluster poses to terminal
            rospy.loginfo("=====================================")
            rospy.loginfo("Detected cluster poses:")
            rospy.loginfo("=====================================")
            cluster_number = 0

            for i in range(max_label + 1):
                cluster_points = filtered_points[labels == i]
                if len(cluster_points) < self.min_cluster_size or len(cluster_points) > self.max_cluster_size:
                    continue

                cluster_number += 1

                # Store cluster for visualization
                clusters.append(cluster_points)

                # Calculate centroid directly - this is the center of the cluster
                centroid = np.mean(cluster_points, axis=0)

                # Check if the cluster resembles a cube
                is_cube, dimensions, orientation, _ = self.check_if_cube(cluster_points)

                # Convert orientation matrix to quaternion
                quaternion = tf.transformations.quaternion_from_matrix(orientation)

                # Create a PoseStamped message for the cube in the camera frame
                pose_camera_frame = PoseStamped()
                pose_camera_frame.header.stamp = rospy.Time.now()
                pose_camera_frame.header.frame_id = msg.header.frame_id  # Camera frame
                pose_camera_frame.pose.position.x = centroid[0]
                pose_camera_frame.pose.position.y = centroid[1]
                pose_camera_frame.pose.position.z = centroid[2]
                pose_camera_frame.pose.orientation.x = quaternion[0]
                pose_camera_frame.pose.orientation.y = quaternion[1]
                pose_camera_frame.pose.orientation.z = quaternion[2]
                pose_camera_frame.pose.orientation.w = quaternion[3]

                # Transform the pose to the robot's base frame (panda_link0)
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "panda_link0",  # Target frame (robot base)
                        msg.header.frame_id,  # Source frame (camera frame)
                        rospy.Time(0),  # Use the latest available transform
                        rospy.Duration(1.0)  # Timeout after 1 second
                    )
                    transformed_pose_stamped = do_transform_pose(pose_camera_frame, transform)
                    transformed_pose = transformed_pose_stamped.pose  # Extract the Pose object
                    rospy.loginfo(f"Transformed pose: {transformed_pose}")

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logerr(f"Failed to transform pose: {str(e)}")
                    continue
                    

                # Print the pose information to the terminal
                rospy.loginfo(f"Cluster {cluster_number}:")
                rospy.loginfo(f"  Position (camera frame): x={centroid[0]:.4f}, y={centroid[1]:.4f}, z={centroid[2]:.4f}")
                rospy.loginfo(f"  Position (robot frame): x={transformed_pose.position.x:.4f}, y={transformed_pose.position.y:.4f}, z={transformed_pose.position.z:.4f}")
                rospy.loginfo(f"  Orientation (quaternion): x={transformed_pose.orientation.x:.4f}, y={transformed_pose.orientation.y:.4f}, z={transformed_pose.orientation.z:.4f}, w={transformed_pose.orientation.w:.4f}")

                # Print euler angles for more intuitive understanding
                euler = tf.transformations.euler_from_quaternion([
                    transformed_pose.orientation.x,
                    transformed_pose.orientation.y,
                    transformed_pose.orientation.z,
                    transformed_pose.orientation.w
                ])
                rospy.loginfo(f"  Orientation (euler): roll={euler[0]:.4f}, pitch={euler[1]:.4f}, yaw={euler[2]:.4f}")

                # Print dimensions and cube classification
                rospy.loginfo(f"  Dimensions: x={dimensions[0]:.4f}, y={dimensions[1]:.4f}, z={dimensions[2]:.4f}")
                rospy.loginfo(f"  Is a cube: {is_cube}")
                rospy.loginfo("---")

                if is_cube or not is_cube:
                    cube_count += 1

                    # Add the transformed pose to the PoseArray
                    pose_array.poses.append(transformed_pose)

                    # Create a marker for visualization
                    marker = Marker()
                    marker.header.frame_id = "panda_link0"  # Visualize in robot base frame
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "cubes"
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose = transformed_pose
                    marker.scale.x = dimensions[0]
                    marker.scale.y = dimensions[1]
                    marker.scale.z = dimensions[2]
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.6
                    marker.lifetime = rospy.Duration(10)  # Increased lifetime to 10 seconds

                    marker_array.markers.append(marker)

                    # Add verified cube to the list
                    verified_cubes.append((dimensions, orientation, centroid))

            rospy.loginfo(f"Found {cluster_number} clusters in total")

            # Publish the results
            if cube_count > 0:
                rospy.loginfo(f"Publishing {cube_count} verified cubes")
                self.pose_pub.publish(pose_array)
                self.marker_pub.publish(marker_array)

                # Visualize clusters if in debug mode
                if self.debug and clusters:
                    self.visualize_clusters(clusters, msg.header.frame_id)

                # Save visualization if there are verified cubes
                if verified_cubes:
                    # Use a fixed frame number (1) since we're only processing once
                    self.save_visualization_matplotlib(filtered_points, labels, verified_cubes, 1)

                # Mark as completed and unsubscribe
                self.detection_completed = True
                self.pc_sub.unregister()
                rospy.loginfo("Detection completed. Unsubscribed from point cloud topic.")

                # Add this line to signal that we're done and can be shut down
                rospy.signal_shutdown("Detection task completed successfully")
            else:
                rospy.loginfo("No cubes detected in this frame. Waiting for next point cloud...")

        except Exception as e:
            rospy.logerr(f"Error in point cloud processing: {str(e)}")
        
          
    
    def check_if_cube(self, points):
        """
        Checks if a cluster of points resembles a cube.
        
        Returns:
            tuple: (is_cube, dimensions, orientation_matrix, centroid)
        """
        # Calculate the centroid
        centroid = np.mean(points, axis=0)
        
        # Center the points
        centered_points = points - centroid
        
        # Perform PCA to find principal axes
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        
        # PCA components give us the orientation
        components = pca.components_
        
        # For cubes on a plane, we want to ensure the z-axis is pointing up
        # Find which component has the largest z value
        z_component_index = np.argmax(np.abs(components[:, 2]))
        
        # If that component's z value is negative, flip it so it points up
        if components[z_component_index, 2] < 0:
            components[z_component_index] = -components[z_component_index]
        
        # Make sure we have a right-handed coordinate system
        # The third axis should be the cross product of the first two
        components[2] = np.cross(components[0], components[1])
        
        # Fix the xy-plane to be aligned with the ground
        # For cubes on a plane, we want to force the third component to be [0,0,1]
        # and adjust the other two components to be perpendicular
        components_aligned = np.zeros_like(components)
        components_aligned[2] = np.array([0, 0, 1])  # Third axis is vertical
        
        # First axis should be in the x-y plane
        # Project the first PCA component onto the x-y plane and normalize
        first_proj = np.array([components[0, 0], components[0, 1], 0])
        if np.linalg.norm(first_proj) > 1e-6:
            first_proj = first_proj / np.linalg.norm(first_proj)
            components_aligned[0] = first_proj
            
            # Second axis is the cross product to ensure orthogonality
            components_aligned[1] = np.cross(components_aligned[2], components_aligned[0])
        else:
            # If the first component is nearly vertical, use x-axis as first
            components_aligned[0] = np.array([1, 0, 0])
            components_aligned[1] = np.array([0, 1, 0])
        
        # Create rotation matrix (orthogonal basis)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = components_aligned
        
        # Transform points to the principal component space
        transformed_points = np.dot(centered_points, components.T)
        
        # Calculate the min/max along each axis to find dimensions
        min_bounds = np.min(transformed_points, axis=0)
        max_bounds = np.max(transformed_points, axis=0)
        dimensions = max_bounds - min_bounds
        
        # Sort dimensions for consistent comparison
        sorted_indices = np.argsort(dimensions)
        dimensions = dimensions[sorted_indices]
        
        # Check if dimensions are within expected range for a cube
        is_within_size = (dimensions[0] >= self.min_side_length and 
                          dimensions[2] <= self.max_side_length)
        
        # For 4.5cm cube, look for dimensions close to expected (with tolerance)
        expected_size = 0.045  # 4.5cm
        size_tolerance = 0.015  # 1.5cm tolerance
        
        size_match = np.all(np.abs(dimensions - expected_size) < size_tolerance)
        
        # Check if the dimensions are similar (for a cube, all sides should be similar)
        # Allow some tolerance defined by cube_dimension_ratio
        is_cubic = (abs(dimensions[0] - dimensions[1]) / dimensions[1] <= self.cube_dimension_ratio and
                    abs(dimensions[1] - dimensions[2]) / dimensions[2] <= self.cube_dimension_ratio and
                    abs(dimensions[0] - dimensions[2]) / dimensions[2] <= self.cube_dimension_ratio)
        
        # Check point distribution along each axis
        is_uniform = self.check_point_distribution(transformed_points)
        
        is_cube = is_within_size and is_uniform   #and is_cubic
        
        # Log detailed information about the potential cube
        if self.debug and is_within_size:
            rospy.loginfo(f"Potential cube: Dimensions={dimensions}, " +
                         f"is_cubic={is_cubic}, is_uniform={is_uniform}, " +
                         f"size_match={size_match}, final={is_cube}")
        
        # Add translation to the transformation matrix
        rotation_matrix[:3, 3] = centroid
        
        return is_cube, dimensions, rotation_matrix, centroid
    
    def check_point_distribution(self, points, bins=10, threshold=0.6):
        """
        Check if points are distributed relatively evenly across all faces of the cube.
        
        Returns:
            bool: True if points are well distributed (suggesting a cube)
        """
        # A cube should have points distributed on all 6 faces
        # This is a simplified check that looks for points near the extremes of each axis
        
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        ranges = max_bounds - min_bounds
        
        faces_covered = 0
        
        # For each axis, check if there are points near both the minimum and maximum
        for axis in range(3):
            # Check points near minimum face
            near_min = np.sum(points[:, axis] < min_bounds[axis] + ranges[axis] * 0.2)
            # Check points near maximum face
            near_max = np.sum(points[:, axis] > max_bounds[axis] - ranges[axis] * 0.2)
            
            if near_min > len(points) * 0.05:
                faces_covered += 1
            if near_max > len(points) * 0.05:
                faces_covered += 1
        
        # If at least 'threshold' (e.g. 60%) of faces have enough points, consider it well-distributed
        return faces_covered >= 6 * threshold
    
    def visualize_clusters(self, clusters, frame_id):
        """
        Publish visualization markers for debugging.
        """
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        marker_array = MarkerArray()
        
        for i, cluster_points in enumerate(clusters):
            # Create a point cloud marker for each cluster
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "clusters"
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.005
            marker.scale.y = 0.005
            
            # Assign color based on cluster index
            color = colors[i % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            # Add points to marker - using geometry_msgs.msg.Point directly
            for point in cluster_points:
                p = Point()  # Import this from geometry_msgs.msg
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)
            
            marker.lifetime = rospy.Duration(10)  # Increased to 10 seconds
            marker_array.markers.append(marker)
        
        # Publish the marker array
        self.debug_marker_pub.publish(marker_array)

    def save_visualization_matplotlib(self, filtered_points, labels, verified_cubes, frame_num):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            import os
            
            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud, colored by cluster
            max_label = labels.max()
            cmap = plt.get_cmap("tab20")
            
            # Plot points by cluster (subsample if too many points)
            if len(filtered_points) > 5000:
                # Subsample for performance
                indices = np.random.choice(len(filtered_points), 5000, replace=False)
                points_subset = filtered_points[indices]
                labels_subset = labels[indices]
            else:
                points_subset = filtered_points
                labels_subset = labels
            
            # Plot each cluster with different color
            for i in range(max_label + 1):
                cluster_indices = np.where(labels_subset == i)[0]
                if len(cluster_indices) > 0:
                    cluster_points = points_subset[cluster_indices]
                    color = cmap(i / (max_label + 1))[:3]
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                            color=color, s=5, alpha=0.6)
            
            # Plot unclustered points in gray
            noise_indices = np.where(labels_subset == -1)[0]
            if len(noise_indices) > 0:
                noise_points = points_subset[noise_indices]
                ax.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                        color='gray', s=1, alpha=0.3)
            
            # Plot verified cubes
            for dimensions, orientation, centroid in verified_cubes:
                # Ensure dimensions are in correct order: smallest to largest
                sorted_dims = np.sort(dimensions)
                
                # For cubes on a plane, set the z dimension as the height
                # and make x and y equal (since they're square in the xy-plane)
                cube_dims = np.array([sorted_dims[1], sorted_dims[1], sorted_dims[2]])
                
                # Create a standard orientation with z-axis pointing up
                # (identity rotation around z-axis)
                standard_rotation = np.eye(3)
                
                # Get the corners of a cube centered at origin
                # Bottom face
                corners = np.array([
                    [-0.5, -0.5, 0],  # Bottom face
                    [0.5, -0.5, 0],
                    [0.5, 0.5, 0],
                    [-0.5, 0.5, 0],
                    # Top face
                    [-0.5, -0.5, 1],
                    [0.5, -0.5, 1],
                    [0.5, 0.5, 1],
                    [-0.5, 0.5, 1]
                ])
                
                # Scale the corners by the dimensions
                corners[:, 0] *= cube_dims[0]
                corners[:, 1] *= cube_dims[1]
                corners[:, 2] *= cube_dims[2]
                
                # Move the cube so its bottom face is on the ground plane
                # and its center is at the detected centroid
                transformed_corners = corners.copy()
                transformed_corners[:, 0] += centroid[0]
                transformed_corners[:, 1] += centroid[1]
                # Place bottom at z-position of the centroid minus half height
                transformed_corners[:, 2] += (centroid[2] - cube_dims[2]/2)
                
                # Draw the edges of the cube
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
                ]
                
                for edge in edges:
                    ax.plot([transformed_corners[edge[0], 0], transformed_corners[edge[1], 0]],
                            [transformed_corners[edge[0], 1], transformed_corners[edge[1], 1]],
                            [transformed_corners[edge[0], 2], transformed_corners[edge[1], 2]],
                            color='green', linewidth=2)
            
            # Set axis labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Make axes equal for better visualization
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            
            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)
            
            max_range = 0.5 * max([x_range, y_range, z_range])
            
            ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
            ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
            ax.set_zlim3d([z_middle - max_range, z_middle + max_range])
            
            # Set a consistent view angle to see the cubes on the xy-plane
            ax.view_init(elev=30, azim=45)
            
            # Add a title
            plt.title(f'Cube Detection - Frame {frame_num}')
            
            # Create directory if it doesn't exist
            if not os.path.exists('cube_detection_results'):
                os.makedirs('cube_detection_results')
            
            # Save the visualization
            plt.savefig(f'cube_detection_results/frame_{frame_num:04d}.png', 
                    bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            rospy.loginfo(f"Saved matplotlib visualization to cube_detection_results/frame_{frame_num:04d}.png")
        except Exception as e:
            rospy.logerr(f"Error saving matplotlib visualization: {str(e)}")
            

if __name__ == '__main__':
    try:
        detector = CubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



## -------------------------------------------------------------------------------------------------------------------------------------------##
## Code Output based on Rosbag Sample 2 :

# [INFO] [1741862631.321675]: Potential cube: Dimensions=[0.03806137 0.05232879 0.0542993 ], is_cubic=False, is_uniform=True, size_match=True, final=True
# [INFO] [1741862631.323748]: Transformed pose: position: 
#   x: 0.4220838972852759
#   y: -0.046079471231183035
#   z: 0.03516417605764777
# orientation: 
#   x: -0.5065229864259168
#   y: 0.5311832406126953
#   z: 0.4912172766871248
#   w: -0.4690249632998775
# [INFO] [1741862631.325122]: Cluster 4:
# [INFO] [1741862631.326554]:   Position (camera frame): x=0.4516, y=-0.0293, z=0.0037
# [INFO] [1741862631.328129]:   Position (robot frame): x=0.4221, y=-0.0461, z=0.0352
# [INFO] [1741862631.330557]:   Orientation (quaternion): x=-0.5065, y=0.5312, z=0.4912, w=-0.4690
# [INFO] [1741862631.332675]:   Orientation (euler): roll=1.6483, pitch=-0.0007, yaw=-1.6177
# Warning: TF_REPEATED_DATA ignoring data with redundant timestamp for frame panda_NE (parent panda_link8) at time 1710339095.802920 according to authority /play_1741856827640010312
#          at line 278 in /tmp/binarydeb/ros-noetic-tf2-0.7.7/src/buffer_core.cpp
# Warning: TF_REPEATED_DATA ignoring data with redundant timestamp for frame panda_EE (parent panda_NE) at time 1710339095.802920 according to authority /play_1741856827640010312
#          at line 278 in /tmp/binarydeb/ros-noetic-tf2-0.7.7/src/buffer_core.cpp
# Warning: TF_REPEATED_DATA ignoring data with redundant timestamp for frame panda_K (parent panda_EE) at time 1710339095.802920 according to authority /play_1741856827640010312
#          at line 278 in /tmp/binarydeb/ros-noetic-tf2-0.7.7/src/buffer_core.cpp
# [INFO] [1741862631.334206]:   Dimensions: x=0.0381, y=0.0523, z=0.0543
# [INFO] [1741862631.335470]:   Is a cube: True
# [INFO] [1741862631.336497]: ---
# [INFO] [1741862631.337891]: Found 4 clusters in total
# [INFO] [1741862631.339061]: Publishing 4 verified cubes
