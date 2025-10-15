-----------
# Franka Zed Gazebo Simulation Project
----------

![Gazebo demo](https://github.com/Ragini313/Intelligent-Robot-Manipulation-/blob/main/digit%20pictures/3D%20Models/robot%20moving%20to%20grab%20cube.gif)

![3_tier_pyramid](https://github.com/Ragini313/Intelligent-Robot-Manipulation-/blob/main/digit%20pictures/3D%20Models/3%20tier%20pyramid_3.gif)

## Overview
This project involves setting up and working with a Gazebo simulation of the Franka Emika Panda robot equipped with a ZED2 camera for educational purposes, focused on pick-and-place tasks and pyramid construction. The simulation is part of the Pearl Robot Lab's educational resources.

## Project Description
The repository contains all necessary files to:
- Launch a Gazebo simulation of the Franka Panda robot
- Integrate a ZED2 camera on the end effector
- Spawn and manipulate cubes in the simulation
- Interface with MoveIt! for motion planning
- Build various pyramid structures with detected cubes
- Work with both simulated and real robot configurations

## Key Features
- Docker setup for easy environment configuration
- Gazebo simulation with Rviz and MoveIt! integration
- Configurable cube spawning system
- ZED2 camera mounted on end-effector for object detection
- Real robot integration capabilities
- Pyramid construction pipeline with multiple configurations:
  - 1-tier to 5-tier pyramids
  - Various tower configurations
- Sample rosbags for offline testing

## Pyramid Construction Pipeline
The system implements a complete pipeline to:
1. Detect randomly placed cubes using the ZED2 camera
2. Plan and execute pick-and-place operations
3. Construct various tower configurations:
   - **Single Tower**: Cubes stacked vertically
   - **Pyramid Structures**:
     - 1-tier: Stacking of user-defined cubes on top of each other
     - 2-tier: Base of 2 cubes + 1 cube on top
     - 3-tier: Base of 3 cubes + 2 cubes + 1 top cube
     - 4-tier: Base of 4 cubes + 3 + 2 + 1 top cube
     - 5-tier: Base of 5 cubes + 4 + 3 + 2 + 1 top cube

The pipeline was first developed and tested in Gazebo simulation before being successfully implemented on the real Franka Emika Panda robot in the PEARL Lab.

## Setup Instructions

### Docker Installation
1. Install Docker following the [PEARL Lab instructions](https://github.com/pearl-robot-lab/Docker_env)
2. To launch the simulation, you will need a few packages including libfranka, franka_ros, MoveIt! and panda_moveit_config. For convenience, you can install a docker and pull images with the necessary packages.
Then you can pull the images.
3. Pull the appropriate Docker image based on your hardware:
   - With NVIDIA GPU: `docker pull 3liyounes/pearl_robots:franka`
   - Without NVIDIA GPU: `docker pull 3liyounes/pearl_robots:franka_wo_nvidia`
   - For real robot testing: `docker pull 3liyounes/pearl_robots:franka_real`
4. Allow X11 access: `xhost +`
5. Create and enter the container:
   ```bash
   source ~/.bashrc
   docker_run_nvidia --name=container_name 3liyounes/pearl_robots:image_name bash
6. Now you are within the docker workspace. If you want to access the workspace in a new terminal, just run:
   ```bash
    docker exec -it container_name bash
7. Get up-to-date repo by
    ```bash
    cd src/franka_zed_gazebo/ && git pull && cd ../..


## To Run our Pipeline
1. Create a package inside your source folder in the franka_zed_gazebo workspace of your container:
   ```bash
   cd ~/franka_zed_gazebo_ws/src
   catkin_create_pkg pyramid_pipeline std_msgs rospy moveit_core
2. Clone this repo into the src folder
   ```bash
   git clone https://github.com/Ragini313/Intelligent-Robot-Manipulation-.git src_LRSY
3. Build the workspace
   ```bash
   cd ~/franka_zed_gazebo_ws
   catkin_make
   source devel/setup.bash
4. Run the pipeline with your desired pyramid configuration.
   ```bash
   rosrun <package_name> pose_detect.py               ## Terminal 1
   rosrun <package_name> stack_object.py              ## Terminal 2

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 For more details, visit the [project repository](https://github.com/pearl-robot-lab/franka_zed_gazebo).
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------




## Authors
- **Ragini Pawar**  
- **Lars Waßmann**

TU Darmstadt WS 2024/25  
[PEARL Lab](https://github.com/pearl-robot-lab)
