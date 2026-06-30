"""
bridge_amr_camera.py — AMR1/AMR2 카메라 ROS2 → 중앙 서버 브리지 노드
================================================================
역할:
  AMR1, AMR2의 카메라 토픽을 구독하여 base64로 인코딩 후
  중앙 Django 서버의 HTTP API로 전달한다.
  시스템 모니터에서 GET /api/amr1/frame/latest/ 로 수신하여 표시.

구독 토픽 → 전송 API:
  /robot2/oakd/rgb/image_raw/compressed → POST /api/amr1/frame/
  /robot4/oakd/rgb/image_raw/compressed → POST /api/amr2/frame/

실행 방법:
  source /opt/ros/humble/setup.bash
  python3 bridge_amr_camera.py
"""

import base64
import threading

import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

SERVER = 'http://192.168.107.42:8000'

AMR1_TOPIC = '/robot2/oakd/rgb/image_raw/compressed'
AMR2_TOPIC = '/robot4/oakd/rgb/image_raw/compressed'


def _post_frame(url, frame_bytes):
    try:
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        requests.post(url, json={'frame': frame_b64}, timeout=1)
    except Exception:
        pass


class AmrCameraBridge(Node):

    def __init__(self):
        super().__init__('amr_camera_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(CompressedImage, AMR1_TOPIC, self._amr1_cb, qos)
        self.create_subscription(CompressedImage, AMR2_TOPIC, self._amr2_cb, qos)

        self.get_logger().info('AMR 카메라 브리지 노드 시작')

    def _amr1_cb(self, msg):
        threading.Thread(
            target=_post_frame,
            args=(f'{SERVER}/api/amr1/frame/', bytes(msg.data)),
            daemon=True,
        ).start()

    def _amr2_cb(self, msg):
        threading.Thread(
            target=_post_frame,
            args=(f'{SERVER}/api/amr2/frame/', bytes(msg.data)),
            daemon=True,
        ).start()


def main():
    rclpy.init()
    node = AmrCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
