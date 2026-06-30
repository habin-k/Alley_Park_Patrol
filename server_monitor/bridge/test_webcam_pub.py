"""
테스트용 웹캠 퍼블리셔
webcam_images/webcam1/detections 토픽으로 카메라 영상을 10fps로 퍼블리시
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = 'webcam_images/webcam1/detections'
CAMERA_INDEX = 0
FPS = 10


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('test_webcam_publisher')
        self.pub = self.create_publisher(Image, TOPIC, 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.get_logger().error(f'웹캠을 열 수 없습니다. (index={CAMERA_INDEX})')
            return
        self.create_timer(1.0 / FPS, self._publish)
        self.get_logger().info(f'퍼블리시 시작 → {TOPIC} ({FPS}fps)')

    def _publish(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
