#!/bin/bash

# 1번 창: Gazebo 월드 실행
gnome-terminal --title="Gazebo World" -- bash -c "
source /opt/ros/humble/setup.bash;

export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/ubean/Robot-Learning/aws-robomaker-small-house-world/models;

ros2 launch gazebo_ros gazebo.launch.py world:=/home/ubean/Robot-Learning/aws-robomaker-small-house-world/worlds/small_house.world;
exec bash"

# Gazebo 로딩을 위해 15초 대기 (컴퓨터 사양에 따라 조절)
sleep 15

# 2번 창: 로봇 소환
gnome-terminal --title="Spawn Robot" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle;
ros2 run gazebo_ros spawn_entity.py -file /opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf -entity turtlebot3_waffle -x 0.0 -y 0.0 -z 0.5;
exec bash"

# 3초 대기
sleep 3

# 3번 창: 키보드 조종
gnome-terminal --title="Teleop Keyboard" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle;
ros2 run turtlebot3_teleop teleop_keyboard;
exec bash"