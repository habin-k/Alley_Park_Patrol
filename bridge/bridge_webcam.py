"""
[프로그램] bridge_webcam.py — 웹캠 ROS2 → 중앙 서버 브리지 노드
================================================================
역할:
  웹캠 팀의 ROS2 토픽을 구독하여 중앙 Django 서버의 HTTP API로 전달한다.
  ROS2와 Django 서버는 서로 다른 통신 방식을 사용하기 때문에,
  이 브리지 노드가 두 시스템 사이의 변환기 역할을 한다.

구독 토픽 → 전송 API:
  webcam_objects/map_detections      (sensor_msgs/Image)
    → POST /api/webcam1/frame/        (바운딩박스 포함 이미지, base64)

  webcam_images/webcam2/detections   (sensor_msgs/Image)
    → POST /api/webcam2/frame/        (웹캠2 이미지, base64)

  webcam_objects/markers             (visualization_msgs/MarkerArray)
    → POST /api/parking/              (차량 맵 좌표 x, y)

실행 방법:
  source /opt/ros/humble/setup.bash
  python3 bridge_webcam.py

주의:
  - ROS2 시스템 Python으로 실행해야 한다 (venv 사용 불가).
  - requests 라이브러리는 시스템 pip으로 설치 필요:
      pip3 install requests
"""

import base64
import threading

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray
from rclpy.node import Node
from sensor_msgs.msg import Image

# 중앙 서버 주소
SERVER = 'http://192.168.107.42:8000'

# JPEG 압축 품질 (0~100). 낮을수록 파일 크기 작아지고 화질 저하.
# 60으로 설정하여 네트워크 부하와 화질의 균형을 맞춤.
JPEG_QUALITY = 60


def _post(url, payload):
    """
    [함수] HTTP POST 요청을 서버로 전송한다.

    Args:
        url     : 요청을 보낼 서버 엔드포인트 URL
        payload : JSON으로 전송할 딕셔너리 데이터

    Note:
        - timeout=1로 설정하여 서버 응답이 없어도 ROS2 루프가 블로킹되지 않도록 함.
        - 예외는 조용히 무시한다 (네트워크 오류 시 로그 스팸 방지).
        - 이 함수는 항상 threading.Thread로 비동기 호출한다.
    """
    try:
        requests.post(url, json=payload, timeout=1)
    except Exception:
        pass


def _encode(cv_img):
    """
    [함수] OpenCV 이미지를 base64 문자열로 인코딩한다.

    Args:
        cv_img : OpenCV BGR 이미지 (numpy ndarray)

    Returns:
        str : JPEG 압축 후 base64 인코딩된 문자열

    Note:
        - JSON으로 이미지를 전송하기 위해 base64 인코딩이 필요하다.
        - JPEG_QUALITY(60)로 압축하여 전송 데이터 크기를 줄인다.
    """
    _, buf = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode('utf-8')


class WebcamBridge(Node):
    """
    [클래스] 웹캠 브리지 ROS2 노드.

    ROS2 토픽 3개를 구독하여 중앙 서버 HTTP API로 데이터를 전달한다.

    Attributes:
        bridge        : CvBridge — ROS2 Image 메시지를 OpenCV 이미지로 변환하는 도구
        _sent_coords  : set — 이미 서버로 전송한 차량 좌표 집합.
                        같은 차량이 매 프레임마다 반복 탐지되는 것을 방지하기 위해 사용.
    """

    def __init__(self):
        """
        [메서드] 노드 초기화.

        - CvBridge 생성
        - 중복 좌표 방지용 set 초기화
        - ROS2 토픽 3개 구독 등록
        """
        super().__init__('webcam_bridge')
        self.bridge = CvBridge()

        # 이미 전송한 좌표를 기억하는 집합 (소수점 1자리로 반올림한 튜플)
        # 예: {(0.6, -1.4), (1.2, 0.3)} → 동일 차량의 중복 전송 방지
        self._sent_coords = set()

        # 웹캠1: 바운딩박스가 그려진 탐지 결과 이미지 → 시스템 모니터 표시용
        self.create_subscription(
            Image,
            'webcam_images/webcam1/detections',
            self._webcam1_cb,
            10,
        )
        # 마커 배열: 탐지된 차량의 맵 좌표 (x, y) → DB 저장용
        self.create_subscription(
            MarkerArray,
            'webcam_objects/map_detections',
            self._map_detections_cb,
            10,
        )

        self.get_logger().info('웹캠 브리지 노드 시작')

    def _webcam1_cb(self, msg):
        """
        [콜백] 웹캠1 이미지 수신 시 호출된다.

        ROS2 Image 메시지를 OpenCV 이미지로 변환하고 base64로 인코딩하여
        서버의 /api/webcam1/frame/ 엔드포인트로 전송한다.
        시스템 모니터 대시보드의 실시간 웹캠1 화면에 표시된다.

        Args:
            msg : sensor_msgs/Image — 웹캠1의 탐지 결과 이미지 메시지
        """
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame_b64 = _encode(frame)
        # 비동기 전송 (메인 ROS2 루프 블로킹 방지)
        threading.Thread(
            target=_post,
            args=(f'{SERVER}/api/webcam1/frame/', {'frame': frame_b64}),
            daemon=True,
        ).start()

    def _map_detections_cb(self, msg):
        """
        [콜백] 차량 탐지 마커 배열 수신 시 호출된다.

        MarkerArray에서 각 차량의 맵 좌표(x, y)를 추출하여
        서버의 /api/parking/ 엔드포인트로 POST 요청을 전송한다.
        서버는 이 좌표를 parking_events 테이블에 저장하고
        AMR1이 이동할 목표 좌표로 사용한다.

        중복 전송 방지 로직:
          1. x == 0.0 또는 y == 0.0 이면 무효 좌표로 스킵
             (탐지 실패 또는 측정 오류로 인해 0이 들어오는 경우)
          2. 이미 전송한 좌표(소수점 1자리 기준)는 재전송 안 함
             (웹캠이 매 프레임마다 같은 차량을 반복 탐지하므로
              차량 1대당 DB row 1개만 생성되도록 보장)

        Args:
            msg : visualization_msgs/MarkerArray — 탐지된 차량의 맵 좌표 마커 배열
        """
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y

            # 무효 좌표 필터링: x 또는 y 중 하나라도 0이면 스킵
            # (둘 다 0인 경우뿐만 아니라 한 축만 0인 경우도 측정 실패로 판단)
            if x == 0.0 or y == 0.0:
                self.get_logger().warn(f'무효 좌표 스킵: ({x:.2f}, {y:.2f})')
                continue

            # 소수점 1자리로 반올림하여 중복 여부 확인
            # 예: x=0.56, y=0.23 → key=(0.6, 0.2) 로 비교
            key = (round(x, 1), round(y, 1))
            if key in self._sent_coords:
                continue  # 이미 전송한 좌표, 스킵

            # 새 좌표 등록 후 서버로 전송
            self._sent_coords.add(key)
            payload = {'observation_x': x, 'observation_y': y}
            threading.Thread(
                target=_post,
                args=(f'{SERVER}/api/parking/', payload),
                daemon=True,
            ).start()
            self.get_logger().info(f'좌표 전송: ({x:.2f}, {y:.2f})')


def main():
    """
    [함수] 노드 진입점.

    rclpy를 초기화하고 WebcamBridge 노드를 생성하여 spin 루프를 실행한다.
    Ctrl+C 입력 시 노드를 안전하게 종료한다.
    """
    rclpy.init()
    node = WebcamBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
