"""
[프로그램] bridge_amr1.py — 중앙 서버 → AMR1 브리지 노드
==========================================================
역할:
  중앙 Django 서버의 HTTP API를 주기적으로 폴링하여
  새로운 주차 탐지 이벤트가 생기면 AMR1의 ROS2 토픽으로 발행한다.

  서버(HTTP) → bridge_amr1.py → ROS2 토픽(webtoamr_xy) → AMR1

동작 흐름:
  1. 2초마다 GET /api/parking/list/ 를 호출하여 전체 이벤트 목록을 조회한다.
  2. status=DETECTED 이면서 아직 발행하지 않은 이벤트만 필터링한다.
  3. 새 이벤트가 있으면 JSON 배열로 직렬화하여 webtoamr_xy 토픽에 발행한다.
  4. AMR1은 이 토픽을 구독하여 Nav2로 자율주행 경로를 계획한다.

발행 토픽:
  webtoamr_xy (std_msgs/String)
  데이터 형식 (JSON 배열):
    [
      {"id": 1, "observation_x": 0.56, "observation_y": 0.23,
       "created_at": "2026-06-25T11:46:39+00:00"},
      ...
    ]

실행 방법:
  source /opt/ros/humble/setup.bash
  python3 bridge_amr1.py

주의:
  - ROS2 시스템 Python으로 실행해야 한다 (venv 사용 불가).
  - requests 라이브러리는 시스템 pip으로 설치 필요:
      pip3 install requests
"""

import os
import json
import requests
import rclpy

os.environ['ROS_DOMAIN_ID'] = '2'
from rclpy.node import Node
from std_msgs.msg import String

# 중앙 서버 주소
SERVER = 'http://192.168.107.42:8000'

# 서버 폴링 간격 (초). 너무 짧으면 서버 부하 증가, 너무 길면 응답 지연.
POLL_INTERVAL = 2.0


class AMR1Bridge(Node):
    """
    [클래스] AMR1 브리지 ROS2 노드.

    서버를 주기적으로 폴링하여 새 DETECTED 이벤트가 생기면
    webtoamr_xy 토픽으로 좌표 정보를 발행한다.

    Attributes:
        _pub          : Publisher — webtoamr_xy 토픽 발행자
        _sent_ids     : set — 이미 발행한 event_id 집합.
                        서버 재시작 전까지 같은 이벤트를 중복 발행하지 않도록 추적.
    """

    def __init__(self):
        """
        [메서드] 노드 초기화.

        - webtoamr_xy 토픽 퍼블리셔 생성
        - 중복 발행 방지용 set 초기화
        - POLL_INTERVAL 주기로 _poll 콜백 타이머 등록
        """
        super().__init__('amr1_bridge')

        # AMR1이 구독할 토픽 퍼블리셔 (std_msgs/String, JSON 문자열)
        self._pub = self.create_publisher(String, 'webtoamr_xy', 10)

        # 이미 발행한 event_id를 기억하는 집합 (중복 발행 방지)
        self._sent_ids = set()

        # POLL_INTERVAL초마다 _poll 메서드를 반복 호출하는 타이머
        self.create_timer(POLL_INTERVAL, self._poll)
        self.get_logger().info('AMR1 브리지 노드 시작')

    def _poll(self):
        """
        [콜백] 타이머 주기(2초)마다 서버를 폴링하여 새 이벤트를 발행한다.

        동작:
          1. GET /api/parking/list/ 로 전체 이벤트 목록 조회
          2. status=DETECTED 이면서 _sent_ids에 없는 이벤트만 선별
          3. 새 이벤트가 있으면 JSON 배열로 직렬화하여 webtoamr_xy 토픽 발행
          4. 발행한 event_id는 _sent_ids에 추가하여 중복 방지

        예외 처리:
          서버가 응답하지 않거나 네트워크 오류 시 경고 로그만 출력하고 다음 주기 대기.
        """
        try:
            res = requests.get(f'{SERVER}/api/parking/list/', timeout=2)
            events = res.json()
        except Exception as e:
            self.get_logger().warn(f'서버 요청 실패: {e}')
            return

        # DETECTED 상태이면서 아직 발행하지 않은 이벤트만 필터링
        new_events = []
        for event in events:
            if event.get('status') != 'DETECTED':
                continue
            event_id = event.get('id')
            if event_id in self._sent_ids:
                continue  # 이미 발행한 이벤트, 스킵

            self._sent_ids.add(event_id)
            new_events.append({
                'id':            event_id,
                'observation_x': event['observation_x'],
                'observation_y': event['observation_y'],
                'created_at':    event['created_at'],
            })

        if not new_events:
            return  # 새 이벤트 없음, 발행 생략

        # JSON 배열로 직렬화하여 토픽 발행
        msg = String()
        msg.data = json.dumps(new_events)
        self._pub.publish(msg)
        self.get_logger().info(
            f'AMR1로 {len(new_events)}개 이벤트 발행: '
            f'ids={[e["id"] for e in new_events]}'
        )


def main():
    """
    [함수] 노드 진입점.

    rclpy를 초기화하고 AMR1Bridge 노드를 생성하여 spin 루프를 실행한다.
    Ctrl+C 입력 시 노드를 안전하게 종료한다.
    """
    rclpy.init()
    node = AMR1Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
