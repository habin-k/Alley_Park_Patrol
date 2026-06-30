"""
bridge_ocr.py — OCR 노드 ↔ Django 서버 브리지
================================================================
역할:
  plate_ocr_node.py의 ROS2 토픽을 구독하여 Django 서버 HTTP API로 전달하고,
  서버 응답을 다시 토픽으로 발행한다.

중요 - event_id 구분:
  plate_ocr_node.py의 event_id = 웹캠이 confidence 순으로 부여한 vehicle_id
  Django DB의 parking_events.id  = DB 자동 생성 PK

  브리지는 vehicle_id → parking_events.id 변환 후 API를 호출한다.
  변환: GET /api/parking/by-vehicle/<vehicle_id>/ → {id, vehicle_id, status}

구독 토픽 → 서버 API:
  regist_car          → POST /api/vehicle/              (AMR1 번호판 등록 + SCANNED)
  request_car         → GET  /api/vehicle/<db_id>/      → servertoocr 발행
  match_result (False)→ DELETE /api/parking/<db_id>/delete/ (불일치 → 이벤트 삭제)
  match_result_id     → POST /api/vehicle/verify/       (일치 → WARNING_ISSUED + 이미지)
  disabled            → GET  /api/disabled/<plate>/     → disabled_result 발행
  disabled_result_id  → POST /api/vehicle/verify/       (장애인 구역 불법주차)
  firecar_result_id   → POST /api/vehicle/verify/       (소방차 구역 불법주차, 무조건 WARNING_ISSUED)

실행 방법:
  source /opt/ros/humble/setup.bash
  python3 bridge_ocr.py
"""

import json
import threading
from urllib.parse import quote

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

SERVER = 'http://192.168.107.42:8000'

# ── 토픽명 (plate_ocr_node.py와 동일) ────────────────────────
REGIST             = 'regist_car'
REQUEST            = 'request_car'
SERVERTOOCR        = 'servertoocr'
MATCH_RESULT       = 'match_result'
MATCH_RESULT_ID    = 'match_result_id'
DISABLED           = 'disabled'
DISABLED_RESULT    = 'disabled_result'
DISABLED_RESULT_ID = 'disabled_result_id'
FIRECAR_RESULT_ID  = 'firecar_result_id'


class OcrBridge(Node):
    """
    OCR 노드와 Django 서버 사이의 브리지 ROS2 노드.

    Attributes:
        _pending_vehicle_id: int | None — request_car 수신 시 저장. match_result(False) 때 사용.
    """

    def __init__(self):
        super().__init__('ocr_bridge')

        self._pending_vehicle_id = None
        self._lock = threading.Lock()

        # Publishers
        self.pub_servertoocr     = self.create_publisher(String, SERVERTOOCR,     10)
        self.pub_disabled_result = self.create_publisher(Bool,   DISABLED_RESULT, 10)

        # Subscribers
        self.create_subscription(String, REGIST,             self.on_regist_car,         10)
        self.create_subscription(String, REQUEST,            self.on_request_car,        10)
        self.create_subscription(Bool,   MATCH_RESULT,       self.on_match_result,       10)
        self.create_subscription(String, MATCH_RESULT_ID,    self.on_match_result_id,    10)
        self.create_subscription(String, DISABLED,           self.on_disabled,           10)
        self.create_subscription(String, DISABLED_RESULT_ID, self.on_disabled_result_id, 10)
        self.create_subscription(String, FIRECAR_RESULT_ID,  self.on_firecar_result_id,  10)

        self.get_logger().info('OCR 브리지 노드 시작')

    # ── 공통 헬퍼: vehicle_id → parking_events.id(DB PK) 변환 ──
    def _get_db_id(self, vehicle_id):
        """
        vehicle_id(웹캠 부여 번호)로 parking_events.id(DB PK)를 조회한다.

        Returns:
            int : parking_events.id. 조회 실패 시 None.
        """
        try:
            resp = requests.get(f'{SERVER}/api/parking/by-vehicle/{vehicle_id}/', timeout=3)
            if resp.status_code == 200:
                return resp.json().get('id')
            self.get_logger().warning(f'DB id 조회 실패: vehicle_id={vehicle_id}, {resp.status_code}')
        except Exception as e:
            self.get_logger().warning(f'DB id 조회 오류: {e}')
        return None

    # ── 1. AMR1 번호판 등록 ──────────────────────────────────────
    def on_regist_car(self, msg):
        """
        regist_car (String) 수신 → POST /api/vehicle/

        payload: {"event_id": vehicle_id, "plate_number": str}
        vehicle_id → DB id 변환 후 서버에 번호판 등록 + status=SCANNED
        """
        data         = json.loads(msg.data)
        vehicle_id   = data['event_id']
        plate_number = data['plate_number']

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'번호판 등록 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.post(f'{SERVER}/api/vehicle/', json={
                    'event_id':     db_id,
                    'plate_number': plate_number,
                }, timeout=3)
                if resp.status_code == 201:
                    self.get_logger().info(
                        f'번호판 등록 성공: vehicle_id={vehicle_id} → db_id={db_id}, plate={plate_number}'
                    )
                else:
                    self.get_logger().warning(f'번호판 등록 실패: {resp.status_code} {resp.text}')
            except Exception as e:
                self.get_logger().warning(f'번호판 등록 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 2. AMR2 번호판 조회 요청 ─────────────────────────────────
    def on_request_car(self, msg):
        """
        request_car (String) 수신 → GET /api/vehicle/<db_id>/ → servertoocr 발행

        payload: {"event_id": vehicle_id}
        vehicle_id → DB id 변환 후 plate_number 조회하여 OCR 노드로 전달.
        match_result(False) 대비로 vehicle_id를 _pending_vehicle_id에 저장.
        """
        data       = json.loads(msg.data)
        vehicle_id = data['event_id']

        with self._lock:
            self._pending_vehicle_id = vehicle_id

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'번호판 조회 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.get(f'{SERVER}/api/vehicle/{db_id}/', timeout=3)
                if resp.status_code == 200:
                    result     = resp.json()
                    reply      = String()
                    reply.data = json.dumps({
                        'event_id':     vehicle_id,
                        'plate_number': result.get('plate_number'),
                    })
                    self.pub_servertoocr.publish(reply)
                    self.get_logger().info(f'번호판 조회 성공: vehicle_id={vehicle_id}')
                else:
                    self.get_logger().warning(f'번호판 조회 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'번호판 조회 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 3. 일반 구역 매칭 결과 (Bool) ────────────────────────────
    def on_match_result(self, msg):
        """
        match_result (Bool) 수신

        True  → match_result_id 토픽에서 이미지와 함께 처리하므로 스킵
        False → vehicle_id → DB id 변환 후 DELETE /api/parking/<db_id>/delete/
        """
        if msg.data:
            return

        with self._lock:
            vehicle_id               = self._pending_vehicle_id
            self._pending_vehicle_id = None

        if vehicle_id is None:
            self.get_logger().warning('match_result False 수신 — pending vehicle_id 없음')
            return

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'이벤트 삭제 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.post(f'{SERVER}/api/vehicle/verify/', json={
                    'event_id': db_id,
                    'match':    False,
                }, timeout=3)
                if resp.status_code == 200:
                    self.get_logger().info(f'불일치 이벤트 삭제 성공: vehicle_id={vehicle_id} → db_id={db_id}')
                else:
                    self.get_logger().warning(f'불일치 이벤트 삭제 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'불일치 이벤트 삭제 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 4. 일반 구역 번호판 일치 (이미지 포함) ────────────────────
    def on_match_result_id(self, msg):
        """
        match_result_id (String) 수신 → POST /api/vehicle/verify/

        payload: {"event_id": vehicle_id, "matched": true, "image": str(base64)}
        vehicle_id → DB id 변환 후 WARNING_ISSUED + ocr_image 저장
        """
        data       = json.loads(msg.data)
        vehicle_id = data['event_id']
        image_b64  = data.get('image')

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'매칭 처리 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.post(f'{SERVER}/api/vehicle/verify/', json={
                    'event_id':  db_id,
                    'match':     True,
                    'ocr_image': image_b64,
                }, timeout=3)
                if resp.status_code == 200:
                    self.get_logger().info(f'번호판 매칭 처리 성공: vehicle_id={vehicle_id} → db_id={db_id}')
                else:
                    self.get_logger().warning(f'번호판 매칭 처리 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'번호판 매칭 처리 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 5. 장애인 구역 - 차량 등록 여부 조회 ─────────────────────
    def on_disabled(self, msg):
        """
        disabled (String) 수신 → GET /api/disabled/<plate>/ → disabled_result (Bool) 발행

        msg.data: 번호판 텍스트 (plain string)
        is_disabled=True  → 등록된 장애인 차량 (정상주차)
        is_disabled=False → 미등록 (불법주차)
        """
        plate_number = msg.data

        def _do():
            try:
                resp = requests.get(f'{SERVER}/api/disabled/{quote(plate_number, safe="")}/', timeout=3)
                if resp.status_code == 200:
                    is_disabled     = resp.json().get('is_disabled', False)
                    result_msg      = Bool()
                    result_msg.data = is_disabled
                    self.pub_disabled_result.publish(result_msg)
                    self.get_logger().info(
                        f'장애인 차량 조회: plate={plate_number}, is_disabled={is_disabled}'
                    )
                else:
                    self.get_logger().warning(f'장애인 차량 조회 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'장애인 차량 조회 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 6. 장애인 구역 불법주차 ───────────────────────────────────
    def on_disabled_result_id(self, msg):
        """
        disabled_result_id (String) 수신 → POST /api/vehicle/verify/

        payload: {"event_id": vehicle_id, "car_number": str, "image": str(base64)}
        vehicle_id → DB id 변환 후 WARNING_ISSUED + ocr_image 저장
        """
        data       = json.loads(msg.data)
        vehicle_id = data['event_id']
        image_b64  = data.get('image')

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'장애인 구역 처리 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.post(f'{SERVER}/api/vehicle/verify/', json={
                    'event_id':  db_id,
                    'match':     True,
                    'ocr_image': image_b64,
                }, timeout=3)
                if resp.status_code == 200:
                    self.get_logger().info(f'장애인 구역 불법주차 처리 성공: vehicle_id={vehicle_id} → db_id={db_id}')
                else:
                    self.get_logger().warning(f'장애인 구역 불법주차 처리 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'장애인 구역 불법주차 처리 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()

    # ── 7. 소방차 구역 불법주차 ──────────────────────────────────
    def on_firecar_result_id(self, msg):
        """
        firecar_result_id (String) 수신 → POST /api/vehicle/verify/

        payload: {"event_id": vehicle_id, "car_number": str, "image": str(base64)}
        vehicle_id → DB id 변환 후 WARNING_ISSUED + ocr_image 저장
        """
        data       = json.loads(msg.data)
        vehicle_id = data['event_id']
        image_b64  = data.get('image')

        def _do():
            db_id = self._get_db_id(vehicle_id)
            if db_id is None:
                self.get_logger().warning(f'소방차 구역 처리 취소: vehicle_id={vehicle_id} DB 레코드 없음')
                return
            try:
                resp = requests.post(f'{SERVER}/api/vehicle/verify/', json={
                    'event_id':  db_id,
                    'match':     True,
                    'ocr_image': image_b64,
                }, timeout=3)
                if resp.status_code == 200:
                    self.get_logger().info(f'소방차 구역 불법주차 처리 성공: vehicle_id={vehicle_id} → db_id={db_id}')
                else:
                    self.get_logger().warning(f'소방차 구역 불법주차 처리 실패: {resp.status_code}')
            except Exception as e:
                self.get_logger().warning(f'소방차 구역 불법주차 처리 오류: {e}')

        threading.Thread(target=_do, daemon=True).start()



def main():
    rclpy.init()
    node = OcrBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
