# vscode 터미널에 아래와 같이 입력하고 나면 촬영가능해진 상태가 됨. 엔터 눌러서 촬영
# source /opt/ros/humble/setup.bash
# python3 manual_capture.py

import os
import cv2
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ManualCapture(Node):
    def __init__(self):
        super().__init__('manual_capture')
        self.bridge = CvBridge()
        self.latest = None
        self.count = 0
        self.out_dir = 'manual_captures'
        os.makedirs(self.out_dir, exist_ok=True)

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        self.latest = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def save_current(self):
        if self.latest is None:
            print('아직 이미지가 안 들어왔습니다.')
            return

        path = os.path.join(self.out_dir, f'capture_{self.count:04d}.png')
        cv2.imwrite(path, self.latest)
        print(f'저장됨: {path}')
        self.count += 1


def spin_node(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = ManualCapture()

    thread = threading.Thread(target=spin_node, args=(node,), daemon=True)
    thread.start()

    print('Enter: 현재 이미지 저장')
    print('q + Enter: 종료')

    while True:
        cmd = input()
        if cmd.lower() == 'q':
            break
        node.save_current()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
