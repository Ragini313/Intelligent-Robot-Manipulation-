-----------
# Franka Zed Gazebo Simulation Project
----------

## Overview
This project involves setting up and working with a Gazebo simulation of the Franka Emika Panda robot equipped with a ZED2 camera for educational purposes, particularly focused on pick-and-place tasks. The simulation is part of the Pearl Robot Lab's educational resources.

## Project Description
The repository contains all necessary files to:
- Launch a Gazebo simulation of the Franka Panda robot
- Integrate a ZED2 camera on the end effector
- Spawn and manipulate objects in the simulation
- Interface with MoveIt! for motion planning
- Work with both simulated and real robot configurations

## Key Features
- Docker setup for easy environment configuration
- Gazebo simulation with Rviz and MoveIt! integration
- Configurable object spawning system
- Multiple ZED2 camera mounting options
- Real robot integration capabilities
- Sample rosbags for offline testing


----------------------------------------------------
## Setup Instructions
------------------------------------------------

### Docker Installation
1. Install Docker following the [official instructions](https://docs.docker.com/engine/install/)
2. Pull the appropriate Docker image based on your hardware:
   - With NVIDIA GPU: `docker pull 3liyounes/pearl_robots:franka`
   - Without NVIDIA GPU: `docker pull 3liyounes/pearl_robots:franka_wo_nvidia`
   - For real robot testing: `docker pull 3liyounes/pearl_robots:franka_real`
3. Allow X11 access: `xhost +`
4. Create and enter the container:
   ```bash
   source ~/.bashrc
   docker_run_nvidia --name=container_name 3liyounes/pearl_robots:image_name bash


TU Darmstadt WS 2024/25
PEARL Lab
