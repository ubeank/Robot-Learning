#!/bin/bash

WORLD_DIR="$HOME/workspace/aws-robomaker-small-house-world"

# 1번 창: Gazebo 월드 실행
gnome-terminal --title="Gazebo World" -- bash -c "
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$WORLD_DIR/models
ros2 launch gazebo_ros gazebo.launch.py world:=$WORLD_DIR/worlds/small_house.world
exec bash
"

sleep 15

# 2번 창: TurtleBot3 소환
gnome-terminal --title="Spawn Robot" -- bash -c "
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 run gazebo_ros spawn_entity.py \
-file /opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf \
-entity turtlebot3_waffle \
-x 0.0 -y 0.0 -z 0.5
exec bash
"

sleep 3

# 3번 창: 키보드 조종
gnome-terminal --title="Teleop Keyboard" -- bash -c "
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 run turtlebot3_teleop teleop_keyboard
exec bash
"

sleep 2

# 4번 창: 1인칭 카메라 뷰
gnome-terminal --title="First Person Camera" -- bash -c "
source /opt/ros/humble/setup.bash
ros2 topic list | grep image
ros2 run rqt_image_view rqt_image_view
exec bash
"

sleep 2

# 5번 창: SLAM 실행
gnome-terminal --title="SLAM Toolbox" -- bash -c "
source /opt/ros/humble/setup.bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true
exec bash
"

sleep 2

# 6번 창: RViz 실행
gnome-terminal --title="RViz" -- bash -c "
source /opt/ros/humble/setup.bash
rviz2
exec bash
"
