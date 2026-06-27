"""
[프로그램] bridge_amr2.py — OCR ↔ 중앙 서버 브리지 노드
=========================================================
역할:
  AMR1/AMR2 OCR 노드와 중앙 Django 서버 사이의 중계 역할을 한다.

구독 토픽 → 서버 API:
  regist_car  (std_msgs/String, JSON)  ← AMR1 OCR
    → POST /api/vehicle/               번호판 등록 → status=SCANNED

  request_car (std_msgs/String, JSON)  ← AMR2 OCR
    → GET /api/monitor/events/         event_id로 plate_number 조회
    → 발행: servertoocr

  ocrtoserver (std_msgs/String, JSON)  ← AMR2 OCR, matched=True
    → POST /api/vehicle/verify/        status → WARNING_ISSUED + 이미지 저장

  ocr_result  (std_msgs/String, JSON)  ← AMR2 OCR, matched=False
    → POST /api/vehicle/verify/        이벤트 삭제

발행 토픽:
  servertoocr (std_msgs/String, JSON)  → AMR2 OCR
    {"event_id": 1, "plate_number": "12가3456"}

실행 방법:
  source /opt/ros/humble/setup.bash
  python3 bridge_amr2.py

주의:
  - ROS2 시스템 Python으로 실행해야 한다 (venv 사용 불가).
  - requests 라이브러리는 시스템 pip으로 설치 필요:
      pip3 install requests
  - 메시지 타입(std_msgs/String)은 추후 커스텀 메시지로 교체 가능.
"""

import os
import json
import threading

import requests
import rclpy

os.environ['ROS_DOMAIN_ID'] = '2'
from rclpy.node import Node
from std_msgs.msg import String

SERVER = 'http://192.168.107.42:8000'


def _post(url, payload, logger=None):
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        if logger:
            logger.warn(f'POST 실패 {url}: {e}')


def _get(url, logger=None):
    try:
        return requests.get(url, timeout=3).json()
    except Exception as e:
        if logger:
            logger.warn(f'GET 실패 {url}: {e}')
        return None


class AMR2Bridge(Node):
    """
    OCR 브리지 ROS2 노드.

    Attributes:
        _servertoocr_pub : Publisher — AMR2 OCR에 번호판 텍스트를 전달하는 토픽
    """

    def __init__(self):
        super().__init__('amr2_bridge')

        # 발행: 서버 → AMR2 OCR (번호판 텍스트 응답)
        self._servertoocr_pub = self.create_publisher(String, 'servertoocr', 10)

        # 구독: AMR1 OCR → 서버 (번호판 등록)
        self.create_subscription(String, 'regist_car',  self._regist_car_cb,  10)
        # 구독: AMR2 OCR → 서버 (번호판 텍스트 요청)
        self.create_subscription(String, 'request_car', self._request_car_cb, 10)
        # 구독: AMR2 OCR → 서버 (매칭 성공 — 이미지 포함)
        self.create_subscription(String, 'ocrtoserver', self._ocrtoserver_cb, 10)
        # 구독: AMR2 OCR → 서버 (매칭 실패 — 삭제 요청)
        self.create_subscription(String, 'ocr_result',  self._ocr_result_cb,  10)

        self.get_logger().info('AMR2 브리지 노드 시작')

    # ------------------------------------------------------------------
    # regist_car: AMR1 OCR → 서버  (번호판 등록 → SCANNED)
    # ------------------------------------------------------------------
    def _regist_car_cb(self, msg):
        """
        기대 JSON:
          {"event_id": 1, "plate_number": "12가3456",
           "amr_vehicle_x": 1.23, "amr_vehicle_y": -0.45}
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'regist_car 파싱 실패: {msg.data[:80]}')
            return

        payload = {
            'event_id':     data.get('event_id'),
            'plate_number': data.get('plate_number'),
            'amr_vehicle_x': data.get('amr_vehicle_x'),
            'amr_vehicle_y': data.get('amr_vehicle_y'),
        }
        threading.Thread(
            target=_post,
            args=(f'{SERVER}/api/vehicle/', payload, self.get_logger()),
            daemon=True,
        ).start()
        self.get_logger().info(
            f'번호판 등록: event_id={payload["event_id"]}, plate={payload["plate_number"]}'
        )

    # ------------------------------------------------------------------
    # request_car: AMR2 OCR → 서버  (번호판 텍스트 요청)
    # servertoocr: 서버 → AMR2 OCR (번호판 텍스트 응답)
    # ------------------------------------------------------------------
    def _request_car_cb(self, msg):
        """
        기대 JSON:
          {"event_id": 1}
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'request_car 파싱 실패: {msg.data[:80]}')
            return

        event_id = data.get('event_id')
        if event_id is None:
            self.get_logger().warn('request_car: event_id 없음')
            return

        threading.Thread(
            target=self._fetch_and_publish_plate,
            args=(event_id,),
            daemon=True,
        ).start()

    def _fetch_and_publish_plate(self, event_id):
        """서버에서 plate_number를 조회하고 servertoocr 토픽으로 발행한다."""
        events = _get(f'{SERVER}/api/monitor/events/?status=SCANNED', self.get_logger())
        if not events:
            self.get_logger().warn(f'SCANNED 이벤트 조회 실패')
            return

        plate_number = None
        for event in events:
            if event.get('id') == event_id:
                vehicle_info = event.get('vehicle_info', [])
                if vehicle_info:
                    plate_number = vehicle_info[0].get('plate_number')
                break

        if plate_number is None:
            self.get_logger().warn(f'event_id={event_id} 번호판 없음')
            return

        out = String()
        out.data = json.dumps({'event_id': event_id, 'plate_number': plate_number})
        self._servertoocr_pub.publish(out)
        self.get_logger().info(f'servertoocr 발행: event_id={event_id}, plate={plate_number}')

    # ------------------------------------------------------------------
    # ocrtoserver: AMR2 OCR → 서버  (matched=True → WARNING_ISSUED)
    # ------------------------------------------------------------------
    def _ocrtoserver_cb(self, msg):
        """
        기대 JSON:
          {"event_id": 1, "image_data": "<base64 string>"}
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'ocrtoserver 파싱 실패: {msg.data[:80]}')
            return

        payload = {
            'event_id':   data.get('event_id'),
            'match':      True,
            'image_data': data.get('image_data'),
        }
        threading.Thread(
            target=_post,
            args=(f'{SERVER}/api/vehicle/verify/', payload, self.get_logger()),
            daemon=True,
        ).start()
        self.get_logger().info(f'매칭 성공 전송: event_id={payload["event_id"]} → WARNING_ISSUED')

    # ------------------------------------------------------------------
    # ocr_result: AMR2 OCR → 서버  (matched=False → 이벤트 삭제)
    # ------------------------------------------------------------------
    def _ocr_result_cb(self, msg):
        """
        기대 JSON:
          {"event_id": 1, "matched": false}
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'ocr_result 파싱 실패: {msg.data[:80]}')
            return

        payload = {
            'event_id': data.get('event_id'),
            'match':    False,
        }
        threading.Thread(
            target=_post,
            args=(f'{SERVER}/api/vehicle/verify/', payload, self.get_logger()),
            daemon=True,
        ).start()
        self.get_logger().info(f'매칭 실패 전송: event_id={payload["event_id"]} → 이벤트 삭제')


def main():
    rclpy.init()
    node = AMR2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
