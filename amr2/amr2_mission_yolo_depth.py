#!/usr/bin/env python3

"""AMR2 mission node.
AMR1과 Webcam에서 들어온 목표를 순서대로 큐에 넣어두고
AMR2가 하나씩 처리함. 하나의 목표가 끝날 때마다 
OCR 결과를 확인하고 경보음을 울린 뒤 다음 목표로 이동
큐가 비게되면 도킹 후 대기 상태로 돌아감
"""

import json
import math
import os
from collections import deque
from typing import Optional
import cv2
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import AudioNote, AudioNoteVector, DockStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool      
from std_msgs.msg import String  
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from ultralytics import YOLO

AMR1_GOAL_TOPIC = '/a_to_b'
IMAGE_TO_OCR = 'target_plate_image'
ID_TO_OCR = 'plate_id'
OCR_RESULT = '/match_result'
DISABLED_RESULT_AMR = '/disabled_result_amr'
WEBCAM_DATA_TOPIC = '/webcam_objects/map_detections'
# OCR Node에는 차량 이미지와 id를 함께 보내야 한다.

NAV_ACTION = '/robot4/navigate_to_pose'
UNDOCK_ACTION = '/robot4/undock'
DOCK_ACTION = '/robot4/dock'

DOCK_STATUS_TOPIC = '/robot4/dock_status'
AUDIO_TOPIC = '/robot4/cmd_audio'
CAMERA_IMAGE_TOPIC = '/robot4/oakd/rgb/preview/image_raw'
PLATE_MODEL_PATH = '/home/dg/Documents/yolo_models/semi_allimages_v5n.pt'

# True이면 언도킹 직후 위치를 "도킹 전에 돌아갈 위치"로 저장한다.
# USE_UNDOCK_POSE_AS_DOCK_POSE = True

# 도킹 스테이션 앞에서 도킹 action을 실행할 위치
DOCK_POSE_X = -0.2
DOCK_POSE_Y = 1.1
DOCK_POSE_YAW = 1.57  # 라디안, 0이면 x축 방향, 1.57이면 y축 방향

OCR_TIMEOUT_SEC = 15.0 # OCR 결과 기다리는 최대 시간
CAMERA_STABILIZE_SEC = 2.5 # 목표 도착 직후 카메라 프레임 안정화 대기 시간
FIRECAR_DELAY_SEC = 2.0 # 소방차 구역 알람 후 다음 목표로 넘어가기 전 대기 시간

ZONE4_GOAL_X = -2.8085745256604517
ZONE4_GOAL_Y = 1.7146872497423458
ZONE4_GOAL_YAW = 2.0939786386670427

ZONE3_GOAL_X = -2.8071492296659484
ZONE3_GOAL_Y = 1.8936850720519807
ZONE3_GOAL_YAW = -2.5405283556901237


class Amr2Mission(Node):

    # 상태 정의.
    # 이 노드는 한 번에 하나의 작업만 해야 하므로 state로 현재 단계를 구분한다.
    IDLE = 'IDLE'
    UNDOCKING = 'UNDOCKING'
    GOING_TO_GOAL = 'GOING_TO_GOAL'
    WAITING_CAMERA_STABLE = 'WAITING_CAMERA_STABLE'
    WAITING_OCR = 'WAITING_OCR'
    WAITING_FIRECAR_DELAY = 'WAITING_FIRECAR_DELAY'
    GOING_TO_DOCK = 'GOING_TO_DOCK'
    DOCKING = 'DOCKING'

    def __init__(self):
        super().__init__('amr2_mission')

        # 아무 작업도 하지 않는 상태
        self.state = self.IDLE 
        # 도킹 상태 받으면 True/False
        self.is_docked: Optional[bool] = None
        # 큐에서 꺼내 처리 중인 목적지. 없으면 None.
        self.goal_pose: Optional[PoseStamped] = None 
        # 현재 목표의 id. 없으면 None.
        self.goal_id: Optional[str] = None
        # 현재 목표로 이동하다 실패한 횟수. 1번만 재시도하고 다음 target으로 넘어간다.
        self.nav_retry_count = 0
        # AMR1/Webcam에서 들어온 목표를 순서대로 처리하기 위한 큐
        self.mission_queue = deque()
        # 현재 처리 중인 큐 항목
        self.current_mission = None
        # 임무가 끝난 뒤 도킹 action 전에 돌아갈 위치
        self.dock_pose: Optional[PoseStamped] = self._dock_pose() 
        # OCR 결과를 기다리는 최대 timeout
        self.camera_stable_deadline_ns: Optional[int] = None
        self.ocr_deadline_ns: Optional[int] = None
        self.firecar_delay_deadline_ns: Optional[int] = None
        # 웹캠에서 받은 map 좌표 목록. 각 항목은 {"id", "zone", "x", "y", "time"}
        self.latest_webcam_targets = []
        self.latest_camera_image: Optional[Image] = None
        self.bridge = CvBridge()

        self.declare_parameter('camera_image_topic', CAMERA_IMAGE_TOPIC)
        self.declare_parameter('plate_model_path', PLATE_MODEL_PATH)
        self.declare_parameter('plate_confidence_threshold', 0.25)
        self.declare_parameter('plate_detection_timeout_sec', 3.0)
        self.declare_parameter('camera_stabilize_sec', CAMERA_STABILIZE_SEC)
        self.declare_parameter('show_detection_window', True)
        self.declare_parameter('detection_window_name', 'AMR2 plate detection')

        self.camera_image_topic = str(
            self.get_parameter('camera_image_topic').value)
        self.plate_model_path = str(
            self.get_parameter('plate_model_path').value)
        self.plate_confidence_threshold = float(
            self.get_parameter('plate_confidence_threshold').value)
        self.plate_detection_timeout_sec = float(
            self.get_parameter('plate_detection_timeout_sec').value)
        self.camera_stabilize_sec = float(
            self.get_parameter('camera_stabilize_sec').value)
        self.show_detection_window = bool(
            self.get_parameter('show_detection_window').value)
        self.detection_window_name = str(
            self.get_parameter('detection_window_name').value)
        self.plate_model = self._load_plate_model(self.plate_model_path)

        # ActionClient
        # send_goal_async()로 요청하고, 결과는 callback에서 받는다.
        self.nav_client = ActionClient(self, NavigateToPose, NAV_ACTION)
        self.undock_client = ActionClient(self, Undock, UNDOCK_ACTION)
        self.dock_client = ActionClient(self, Dock, DOCK_ACTION)

        # AMR1에서 id와 좌표가 들어있는 JSON 문자열 수신.
        self.create_subscription(
            String, AMR1_GOAL_TOPIC, self._goal_callback, 10)
        # OCR 결과 수신.
        self.create_subscription(
            Bool,
            OCR_RESULT,
            lambda msg: self._ocr_callback(msg, {1, 2}, 'match_result'),
            10)
        self.create_subscription(
            Bool,
            DISABLED_RESULT_AMR,
            lambda msg: self._ocr_callback(msg, {3}, 'disabled_result_amr'),
            10)
        # 웹캠 JSON map 좌표 수신.
        self.create_subscription(
            String, WEBCAM_DATA_TOPIC, self._webcam_data_callback, 10)
        # TurtleBot4 도킹 상태 수신.
        self.create_subscription(
            DockStatus,
            DOCK_STATUS_TOPIC,
            self._dock_status_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Image,
            self.camera_image_topic,
            self._camera_image_callback,
            qos_profile_sensor_data)

        # OCR 노드에 검사 요청을 보낼 publisher.
        self.image_to_ocr = self.create_publisher(
            CompressedImage, IMAGE_TO_OCR, 10)
        self.id_to_ocr = self.create_publisher(
            String, ID_TO_OCR, 10)
        
        self.audio_pub = self.create_publisher(
            AudioNoteVector, AUDIO_TOPIC, 10)

        # 0.2초마다 _timer_callback이 실행
        # 이 timer가 전체 임무 흐름을 계속 확인하고 다음 단계를 시작한다.
        self.timer = self.create_timer(0.2, self._timer_callback)

        self.get_logger().info(
            f'AMR2 YOLO mission node ready. camera={self.camera_image_topic}, '
            f'plate_model={self.plate_model_path}')


    def _goal_callback(self, msg: String):
        """AMR1에서 JSON 문자열로 목적지를 받으면 큐에 저장한다."""
        mission = self._amr1_json_to_mission(msg)
        if mission is None:
            return

        self._enqueue_mission(mission, 'AMR1')

    def _amr1_json_to_mission(self, msg: String):
        """AMR1 JSON 문자열에서 event_id, 위치, 방향을 꺼내 큐 항목으로 변환한다."""
        try:
            data = json.loads(msg.data)
            goal_id = str(data['event_id'])

            if 'map' in data:
                point = data['map']
            else:
                point = data

            orientation = point.get('orientation', data.get('orientation'))

            mission = {
                'id': goal_id,
                'zone': data.get('zone'),
                'x': float(point['x']),
                'y': float(point['y']),
                'yaw': 0.0,
            }
            if orientation is not None:
                mission['orientation'] = {
                    'x': float(orientation['x']),
                    'y': float(orientation['y']),
                    'z': float(orientation['z']),
                    'w': float(orientation['w']),
                }
            return mission

        except Exception as e:
            self.get_logger().error(f'AMR1 goal json: {e}')
            return None

    def _webcam_data_callback(self, msg: String):
        """웹캠 노드에서 온 JSON 문자열을 map 좌표 목록으로 변환해 저장한다."""
        targets = self.json_to_dic(msg)
        if not targets:
            return

        self.latest_webcam_targets = targets
        for target in targets:
            mission = self._webcam_target_to_mission(target)
            if mission is not None:
                self._enqueue_mission(mission, 'Webcam')

        self.get_logger().info(
            f'웹캠 map 좌표 수신: {len(self.latest_webcam_targets)}개')

    def _webcam_target_to_mission(self, target):
        """웹캠 target을 zone별 고정 접근 좌표 mission으로 변환한다."""
        if target['zone'] == 3:
            return {
                'id': str(target['id']),
                'zone': target['zone'],
                'x': ZONE3_GOAL_X,
                'y': ZONE3_GOAL_Y,
                'yaw': ZONE3_GOAL_YAW,
            }

        if target['zone'] == 4:
            return {
                'id': str(target['id']),
                'zone': target['zone'],
                'x': ZONE4_GOAL_X,
                'y': ZONE4_GOAL_Y,
                'yaw': ZONE4_GOAL_YAW,
            }

        self.get_logger().warning(f'지원하지 않는 zone 무시: {target["zone"]}')
        return None

    def _enqueue_mission(self, mission, source: str):
        """AMR1/Webcam에서 들어온 목표를 mission queue에 추가한다."""
        mission_id = mission['id']
        if self._mission_id_exists(mission_id):
            self.get_logger().info(f'중복 목표 무시: id={mission_id}')
            return

        self.mission_queue.append(mission)
        self.get_logger().info(
            f'{source} 목표 큐 추가: id={mission_id}, '
            f'zone={mission.get("zone")}, '
            f'x={mission["x"]:.3f}, y={mission["y"]:.3f}, '
            f'queue={len(self.mission_queue)}')

    def _mission_id_exists(self, mission_id: str) -> bool:
        """현재 처리 중이거나 큐에 대기 중인 id인지 확인한다."""
        if self.current_mission is not None:
            if self.current_mission['id'] == mission_id:
                return True
        return any(mission['id'] == mission_id for mission in self.mission_queue)

    def _load_next_mission(self) -> bool:
        """큐에서 다음 목표를 꺼내 현재 goal로 설정한다."""
        if not self.mission_queue:
            return False

        self.current_mission = self.mission_queue.popleft()
        self.goal_id = self.current_mission['id']
        self.goal_pose = self._mission_to_pose(self.current_mission)
        self.nav_retry_count = 0
        self.get_logger().info(
            f'다음 목표 선택: id={self.goal_id}, '
            f'zone={self.current_mission.get("zone")}, '
            f'x={self.goal_pose.pose.position.x:.3f}, '
            f'y={self.goal_pose.pose.position.y:.3f}, '
            f'남은 queue={len(self.mission_queue)}')
        return True

    def _mission_to_pose(self, mission) -> PoseStamped:
        """큐 항목을 Nav2가 받을 PoseStamped로 변환한다."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = mission['x']
        pose.pose.position.y = mission['y']
        if 'orientation' in mission:
            orientation = mission['orientation']
            pose.pose.orientation.x = orientation['x']
            pose.pose.orientation.y = orientation['y']
            pose.pose.orientation.z = orientation['z']
            pose.pose.orientation.w = orientation['w']
        else:
            pose.pose.orientation.z = math.sin(mission['yaw'] / 2.0)
            pose.pose.orientation.w = math.cos(mission['yaw'] / 2.0)
        return self._copy_goal(pose)

    def json_to_dic(self, msg: String):
        """웹캠 JSON 문자열에서 zone3/zone4의 id와 좌표만 꺼낸다."""
        try:
            data = json.loads(msg.data)

            camera_id = data.get('id', 'webcam')
            system_time = data.get('system_time', self.get_clock().now().nanoseconds)
            objects = data.get('objects', data.get('detections', []))

            targets = []
            for index, det in enumerate(objects):
                zone = det.get('zone')
                if zone not in (3, 4):
                    continue

                point = det.get('map', det)
                target_id = det.get(
                    'event_id', det.get('id', f'{camera_id}_{index}'))
                targets.append({
                    'id': str(target_id),
                    'zone': zone,
                    'x': point['x'],
                    'y': point['y'],
                    'time': system_time,
                })

            return targets

        except Exception as e:
            self.get_logger().error(f'json: {e}')
            return []

    def _dock_status_callback(self, msg: DockStatus):
        """로봇이 도킹 중인지 저장한다."""
        old_status = self.is_docked
        self.is_docked = bool(msg.is_docked)
        # 상태가 바뀔 때만 로그를 찍어서 같은 로그가 계속 반복되지 않게 한다.
        if old_status != self.is_docked:
            text = '도킹됨' if self.is_docked else '도킹 해제됨'
            self.get_logger().info(f'도킹 상태: {text}')

    def _camera_image_callback(self, msg: Image):
        """OCR에 보낼 원본 카메라 프레임을 최신값으로 저장한다."""
        self.latest_camera_image = msg

    def _timer_callback(self):
        """현재 상태를 보고 다음 일을 시작한다."""
        # OCR 대기 중일 때는 timeout만 확인하고 다른 행동은 하지 않는다.
        if self.state == self.WAITING_OCR:
            self._check_ocr_timeout()
            return

        if self.state == self.WAITING_CAMERA_STABLE:
            self._check_camera_stable()
            return

        if self.state == self.WAITING_FIRECAR_DELAY:
            self._check_firecar_delay()
            return

        # IDLE이 아니면 이미 action이 진행 중이다.
        # 예: 언도킹 중, 이동 중, 도킹 중.
        if self.state != self.IDLE:
            return

        # 현재 목표가 없으면 큐에서 다음 목표를 꺼낸다.
        if self.goal_pose is None:
            if not self._load_next_mission():
                return

        # 목표가 있고 도킹 상태이면 언도킹부터 한다.
        if self.is_docked is True:
            self._start_undock()
        # 이미 도킹 해제 상태이면 바로 목표로 간다.
        elif self.is_docked is False:
            self._start_go_to_goal()
        else:
            self.get_logger().warning('dock_status를 아직 못 받았습니다.')

    def _start_undock(self):
        """도킹 상태이면 먼저 언도킹"""
        if not self.undock_client.server_is_ready():
            self.get_logger().warning(f'Undock 서버 대기 중: {UNDOCK_ACTION}')
            return

        self.state = self.UNDOCKING
        self.get_logger().info('언도킹 시작')
        future = self.undock_client.send_goal_async(Undock.Goal())
        # goal 요청에 대한 응답이 오면 _undock_goal_response가 실행된다.
        future.add_done_callback(self._undock_goal_response)

    def _undock_goal_response(self, future):
        """언도킹 action server가 요청을 받아줬는지 확인"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('언도킹 goal 거부됨')
            self.state = self.IDLE
            return

        # 요청을 받아줬다면 실제 언도킹이 끝날 때까지 결과를 기다린다.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._undock_result)

    def _undock_result(self, future):
        # 언도킹 action 결과
        if future.result().status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('언도킹 실패')
            self.state = self.IDLE
            return
        
        self.is_docked = False 
        self.state = self.IDLE
        self.get_logger().info('언도킹 완료')

    def _start_go_to_goal(self):
        """AMR1이 준 목적지로 이동한다."""
        if self.goal_pose is None:
            return
        
        if not self.nav_client.server_is_ready():
            self.get_logger().warning(f'Nav2 서버 대기 중: {NAV_ACTION}')
            return

        # Nav2가 오래된 goal이라고 판단하지 않도록 현재 시간으로 stamp를 갱신한다.
        self.goal_pose.header.stamp = self.get_clock().now().to_msg()

        request = NavigateToPose.Goal()
        request.pose = self.goal_pose
        self.state = self.GOING_TO_GOAL
        self.get_logger().info(
            f'목표로 이동 시작: x={request.pose.pose.position.x:.3f}, '
            f'y={request.pose.pose.position.y:.3f}')

        future = self.nav_client.send_goal_async(request)
        future.add_done_callback(self._nav_goal_response)

    def _nav_goal_response(self, future):
        """Nav2가 이동 goal을 받았는지 확인한다."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self.state == self.GOING_TO_GOAL:
                self._handle_goal_navigation_failure(
                    'Nav2가 goal을 거부했습니다.')
            else:
                self.get_logger().error('Nav2가 goal을 거부했습니다.')
                self.state = self.IDLE
            return

        # goal이 수락되면 이동이 끝날 때까지 결과를 기다린다.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result)

    def _handle_goal_navigation_failure(self, reason: str):
        """현재 목표 이동 실패를 처리한다."""
        if self.nav_retry_count < 1:
            self.nav_retry_count += 1
            self.get_logger().warning(
                f'{reason} 1회 재시도: id={self.goal_id}')
            self.state = self.IDLE
            return

        self.get_logger().error(
            f'{reason} 최종 실패. 다음 목표로 넘어갑니다: id={self.goal_id}')
        self._finish_current_mission()

    def _nav_result(self, future):
        """Nav2 이동이 끝났을 때 실행된다."""
        status = future.result().status

        if status != GoalStatus.STATUS_SUCCEEDED:
            if self.state == self.GOING_TO_GOAL and self.nav_retry_count < 1:
                self._handle_goal_navigation_failure(
                    f'Nav2 이동 실패: status={status}')
                return

            if self.state == self.GOING_TO_GOAL:
                self._handle_goal_navigation_failure(
                    f'Nav2 이동 실패: status={status}')
                return

            self.get_logger().error(f'Nav2 이동 실패: status={status}')
            self.state = self.IDLE
            return

        # 같은 Nav2 이동이라도 현재 state에 따라 다음 일이 달라진다.
        if self.state == self.GOING_TO_GOAL:
            # 목적지에 도착했으면 카메라 프레임이 안정될 시간을 둔 뒤 검사한다.
            self._start_camera_stabilize()
        elif self.state == self.GOING_TO_DOCK:
            # 도킹 위치에 도착했으면 실제 dock action을 실행한다.
            self.get_logger().info('도킹 위치 도착')
            self._start_dock()

    def _start_camera_stabilize(self):
        """목표 도착 직후 흔들린 카메라 프레임을 피하기 위해 잠깐 대기한다."""
        self.state = self.WAITING_CAMERA_STABLE
        self.camera_stable_deadline_ns = (
            self.get_clock().now().nanoseconds
            + int(self.camera_stabilize_sec * 1_000_000_000)
        )
        self.get_logger().info(
            f'목표 도착. 카메라 안정화 {self.camera_stabilize_sec:.1f}초 대기')

    def _check_camera_stable(self):
        """카메라 안정화 시간이 지나면 OCR 검사를 시작한다."""
        if self.camera_stable_deadline_ns is None:
            return
        if self.get_clock().now().nanoseconds < self.camera_stable_deadline_ns:
            return

        self.camera_stable_deadline_ns = None
        self._start_ocr_wait()

    def _start_ocr_wait(self):
        """목표 지점 도착 후 OCR 노드에 검사를 요청한다."""
        current_zone = self.current_mission.get('zone')
        img_msg = self._make_target_plate_image(self.current_mission)
        if img_msg is None:
            self.get_logger().warning('차량 이미지 생성 실패. OCR 요청 전송 생략')
        else:
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_to_ocr.publish(img_msg)

            target_msg = String()
            target_msg.data = json.dumps({
                'event_id': self.goal_id,
                'zone': str(current_zone),
            })
            self.id_to_ocr.publish(target_msg)

        if current_zone == 4:
            self.get_logger().info('소방차 구역 도착. 알람 후 2초 대기')
            self._play_alarm('소방차 구역 알람음 발생')
            self.state = self.WAITING_FIRECAR_DELAY
            self.firecar_delay_deadline_ns = (
                self.get_clock().now().nanoseconds
                + int(FIRECAR_DELAY_SEC * 1_000_000_000)
            )
            return

        if img_msg is None:
            self.get_logger().warning('OCR 요청 불가. 현재 목표를 종료합니다.')
            self._finish_current_mission()
            return

        self.state = self.WAITING_OCR
        # 현재 시간 + OCR_TIMEOUT_SEC를 저장한다.
        # timer callback에서 이 시간이 지났는지 계속 확인한다.
        self.ocr_deadline_ns = (
            self.get_clock().now().nanoseconds
            + int(OCR_TIMEOUT_SEC * 1_000_000_000)
        )
        self.get_logger().info('목표 도착. OCR 결과 대기')

    def _make_target_plate_image(self, mission) -> Optional[CompressedImage]:
        """YOLO 번호판 bbox 확인 후 OCR에 보낼 원본 이미지를 반환한다."""
        if mission is None:
            return None

        event_id = mission['id']
        zone = mission.get('zone')
        goal_x = mission['x']
        goal_y = mission['y']

        if self.plate_model is None:
            self.get_logger().error('번호판 YOLO 모델이 로드되지 않았습니다.')
            return None

        frame_msg = self._wait_for_camera_image()
        if frame_msg is None:
            self.get_logger().error('카메라 이미지를 시간 안에 받지 못했습니다.')
            return None

        try:
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().error(f'카메라 Image 변환 실패: {error}')
            return None

        bbox = self._detect_plate_bbox(frame)
        if bbox is None:
            self.get_logger().warning(
                f'번호판 bbox 탐지 실패: id={event_id}, zone={zone}, '
                f'goal=({goal_x:.3f}, {goal_y:.3f})')
            return None

        x1, y1, x2, y2, conf = bbox
        self._show_detection_frame(frame, bbox)
        self.get_logger().info(
            f'번호판 bbox 확인 후 원본 이미지 전송: id={event_id}, zone={zone}, '
            f'conf={conf:.3f}, xyxy=({x1}, {y1}, {x2}, {y2})')
        return self._image_to_compressed(frame_msg, frame)

    def _image_to_compressed(self, frame_msg: Image, frame) -> Optional[CompressedImage]:
        """카메라 원본 프레임을 OCR 노드가 받는 CompressedImage로 변환한다."""
        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            self.get_logger().error('카메라 원본 이미지 JPEG 압축 실패')
            return None

        msg = CompressedImage()
        msg.header = frame_msg.header
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        return msg

    def _show_detection_frame(self, frame, bbox):
        """번호판 탐지 결과를 OpenCV 창으로 표시한다."""
        if not self.show_detection_window:
            return

        x1, y1, x2, y2, conf = bbox
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f'Plate {conf:.2f}',
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA)
        cv2.imshow(self.detection_window_name, display)
        cv2.waitKey(1)

    def _load_plate_model(self, model_path: str):
        """YOLO 번호판 탐지 모델을 로드한다."""
        if not model_path:
            self.get_logger().error('plate_model_path 파라미터가 비어 있습니다.')
            return None
        if not os.path.exists(model_path):
            self.get_logger().error(f'번호판 YOLO 모델 파일 없음: {model_path}')
            return None

        try:
            return YOLO(model_path)
        except Exception as error:
            self.get_logger().error(f'번호판 YOLO 모델 로드 실패: {error}')
            return None

    def _wait_for_camera_image(self) -> Optional[Image]:
        """목표 이동 중 구독해둔 최신 카메라 이미지를 반환한다."""
        _ = self.plate_detection_timeout_sec
        return self.latest_camera_image

    def _detect_plate_bbox(self, frame):
        """YOLO 결과 중 confidence가 가장 높은 bbox 좌표를 반환한다."""
        try:
            results = self.plate_model.predict(
                source=frame,
                conf=self.plate_confidence_threshold,
                verbose=False)
        except Exception as error:
            self.get_logger().error(f'번호판 YOLO 추론 실패: {error}')
            return None

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_box = max(boxes, key=lambda box: float(box.conf[0]))
        conf = float(best_box.conf[0])
        if conf < self.plate_confidence_threshold:
            return None

        x1, y1, x2, y2 = [int(round(value)) for value in best_box.xyxy[0].tolist()]
        height, width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2, conf

    def _ocr_callback(self, msg: Bool, allowed_zones=None, source='OCR'):
        """OCR 결과를 받으면 알람 여부를 처리하고 도킹 위치로 돌아간다."""
        # OCR을 기다리는 상태가 아니면 결과를 무시한다.
        # 예: 이전 임무의 늦은 OCR 결과가 들어오는 경우.
        if self.state != self.WAITING_OCR:
            return

        if self.current_mission is None:
            return

        current_zone = self.current_mission.get('zone')
        if allowed_zones is not None and current_zone not in allowed_zones:
            self.get_logger().info(
                f'{source} 결과 무시: 현재 zone={current_zone}')
            return

        self.get_logger().info(
            f'OCR 결과({source}, zone={current_zone}): {msg.data}')
        # OCR 결과가 True이면 알람음을 낸다.
        if msg.data:
            self._play_alarm()

        self._finish_current_mission()

    def _check_ocr_timeout(self):
        """OCR 결과가 너무 늦으면 현재 임무를 끝내고 다음 단계로 넘어간다."""
        if self.ocr_deadline_ns is None:
            return
        # 아직 제한 시간이 지나지 않았으면 계속 기다린다.
        if self.get_clock().now().nanoseconds < self.ocr_deadline_ns:
            return

        self.get_logger().warning('OCR timeout. 현재 목표를 종료합니다.')
        self._finish_current_mission()

    def _check_firecar_delay(self):
        """소방차 구역 알람 후 잠깐 대기한 뒤 현재 임무를 종료한다."""
        if self.firecar_delay_deadline_ns is None:
            return
        if self.get_clock().now().nanoseconds < self.firecar_delay_deadline_ns:
            return

        self.get_logger().info('소방차 구역 2초 대기 완료. 현재 목표를 종료합니다.')
        self._finish_current_mission()

    def _finish_current_mission(self):
        """현재 목표를 끝내고 큐가 남았으면 다음 목표, 없으면 도킹으로 넘어간다."""
        self.goal_pose = None
        self.goal_id = None
        self.current_mission = None
        self.camera_stable_deadline_ns = None
        self.ocr_deadline_ns = None
        self.firecar_delay_deadline_ns = None

        if self.mission_queue:
            self.state = self.IDLE
            self.get_logger().info(
                f'다음 목표 대기: queue={len(self.mission_queue)}')
            return

        self._start_go_to_dock_pose()

    def _start_go_to_dock_pose(self):
        """도킹 action 전에 충전독 앞 위치로 이동한다."""
        # dock_pose는 하드코딩 좌표이다.
        if self.dock_pose is None:
            self.get_logger().error('도킹 복귀 위치가 없습니다.')
            self.state = self.IDLE
            return
        if not self.nav_client.server_is_ready():
            self.get_logger().warning(f'Nav2 서버 대기 중: {NAV_ACTION}')
            self.state = self.IDLE
            return

        # Nav2 goal로 보내기 전에 stamp를 현재 시간으로 갱신한다.
        self.dock_pose.header.stamp = self.get_clock().now().to_msg()

        request = NavigateToPose.Goal()
        request.pose = self.dock_pose
        self.state = self.GOING_TO_DOCK
        self.get_logger().info(
            f'도킹 위치로 복귀 시작: x={request.pose.pose.position.x:.3f}, '
            f'y={request.pose.pose.position.y:.3f}')

        future = self.nav_client.send_goal_async(request)
        future.add_done_callback(self._nav_goal_response)

    def _start_dock(self):
        """충전독 앞에서 TurtleBot4 dock action을 실행한다."""
        # dock action은 충전독 근처에서만 성공하기 쉽다.
        # 이 함수는 _start_go_to_dock_pose 이후에 호출됨 (즉, 도킹 위치에 도착한 상태)
        if not self.dock_client.server_is_ready():
            self.get_logger().warning(f'Dock 서버 대기 중: {DOCK_ACTION}')
            self.state = self.IDLE
            return

        self.state = self.DOCKING
        self.get_logger().info('도킹 시작')
        future = self.dock_client.send_goal_async(Dock.Goal())
        future.add_done_callback(self._dock_goal_response)

    def _dock_goal_response(self, future):
        """dock action server가 요청을 받아줬는지 확인한다."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('도킹 goal 거부됨')
            self.state = self.IDLE
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._dock_result)

    def _dock_result(self, future):
        """도킹 action의 최종 결과를 처리한다."""
        if future.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.is_docked = True
            self.get_logger().info('도킹 완료. 다음 목표 대기')
        else:
            self.get_logger().error('도킹 실패')
        self.state = self.IDLE

    def _play_alarm(self, log_text='OCR=True: 2초 알람음 발생'):
        """2초 동안 알람음을 낸다."""
        msg = AudioNoteVector()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.append = False

        # 0.5초짜리 음 4개를 이어서 총 2초 알람을 만든다.
        for hz in (880, 660, 880, 660):
            note = AudioNote()
            note.frequency = hz
            note.max_runtime.nanosec = 500_000_000
            msg.notes.append(note)

        self.audio_pub.publish(msg)
        self.get_logger().warning(log_text)

    def _copy_goal(self, source: PoseStamped) -> PoseStamped:
        """받은 goal을 Nav2에 보내기 좋은 형태로 복사한다."""
        copied = self._copy_pose(source)
        # frame_id는 좌표 기준이다. 비어 있으면 map 좌표계로 가정한다.
        if not copied.header.frame_id:
            copied.header.frame_id = 'map'

        # orientation은 quaternion이다.
        # Nav2는 길이가 1인 정상 quaternion을 기대하므로 정규화한다.
        q = copied.pose.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-6:
            # 방향 값이 완전히 비어 있으면 yaw=0과 같은 기본 방향을 넣는다.
            q.w = 1.0
        else:
            q.x /= norm
            q.y /= norm
            q.z /= norm
            q.w /= norm
        return copied

    def _copy_pose(self, source: PoseStamped) -> PoseStamped:
        """PoseStamped를 새 객체로 복사한다.

        ROS 메시지 객체를 그대로 공유하지 않고 복사해두면,
        나중에 stamp 같은 값을 수정해도 원본 callback 메시지와 섞이지 않는다.
        """
        copied = PoseStamped()
        copied.header.frame_id = source.header.frame_id
        copied.header.stamp = source.header.stamp
        copied.pose.position.x = source.pose.position.x
        copied.pose.position.y = source.pose.position.y
        copied.pose.position.z = source.pose.position.z
        copied.pose.orientation.x = source.pose.orientation.x
        copied.pose.orientation.y = source.pose.orientation.y
        copied.pose.orientation.z = source.pose.orientation.z
        copied.pose.orientation.w = source.pose.orientation.w
        return copied

    def _dock_pose(self) -> PoseStamped:
        """파일 위쪽의 DOCK_POSE_X/Y/YAW 값으로 도킹 복귀 위치를 만든다."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = DOCK_POSE_X
        pose.pose.position.y = DOCK_POSE_Y
        pose.pose.orientation.z = math.sin(DOCK_POSE_YAW / 2.0)
        pose.pose.orientation.w = math.cos(DOCK_POSE_YAW / 2.0)
        return pose

def main(args=None):
    rclpy.init(args=args)
    node = Amr2Mission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
