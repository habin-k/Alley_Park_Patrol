import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class MultiWebcamObjectMapper5(Node):
    """Show all detections, but publish only car objects in the JSON payload."""

    NODE_NAME = 'multi_webcam_object_mapper5'
    WINDOW_PREFIX = 'multi_webcam_object_mapper5'
    CAMERA_NAME = 'webcam1'
    ANNOTATED_IMAGE_TOPIC = '/webcam_images/webcam1/detections'
    JSON_CLASS_ID = 0
    FIRE_TRUCK_CLASS_ID = 4
    USE_ILLEGAL_ZONE_FILTER = True
    ILLEGAL_ZONES = [
        {
            # ZONE1 영역 (좌상단부터 시계방향)
            'type': 'polygon',
            'points': [
                [-3.3, -0.23],
                [-3.25, 0.05],
                [-1.38, 0.03],
                [-1.47, -0.319],
            ],
        },
        {
            # ZONE2 영역 (좌상단부터 시계방향)
            'type': 'polygon',
            'points': [
                [-4.19, 0.35],
                [-4.13, 0.964],
                [-1.24, 0.93],
                [-1.3, 0.55],
                [-3.45, 0.55],
                [-3.45, 0.35],
            ],
        },
        {
            # ZONE3 영역 (좌상단부터 시계방향)
            'type': 'polygon',
            'points': [
                [-3.85, 0.95],
                [-3.81, 1.57],
                [-3.1, 1.57],
                [-3.15, 0.95],
            ],
        },
        {
            # ZONE4 영역 (좌상단부터 시계방향)
            'type': 'polygon',
            'points': [
                [-4.27, 2.12],
                [-4.19, 2.8],
                [-3.04, 2.75],
                [-3.1, 2.14],
            ],
        },
    ]

    def __init__(self):
        super().__init__(self.NODE_NAME)					            # ROS 노드 이름 multi_webcam_object_mapper5로 생성
        self._declare_parameters()							            # 노드에서 사용할 ROS 파라미터선언                
        self.model = self._load_yolo_model()                            # YOLO 객체 감지 모델을 불러옵니다.						        
        self.class_names = self._model_class_names()                    # 모델의 클래스 ID와 이름 목록을 가져옵니다.
        self.confidence_threshold = float(						        # 객체 감지 결과를 사용할 최소 신뢰도를 다룹니다.
            self.get_parameter('confidence_threshold').value)
        self.class_ids = self._parse_int_list(						    # 감지할 클래스 ID 필터 목록을 다룹니다.
            self.get_parameter('class_ids').value)
        self.show_image = bool(self.get_parameter('show_image').value)          # 카메라 영상 창 표시 여부를 다룹니다.
        self.display_width = int(self.get_parameter('display_width').value)
        self.display_height = int(self.get_parameter('display_height').value)
        self.window_name = f'{self.WINDOW_PREFIX} {self.CAMERA_NAME}'
        self.window_initialized = False
        self.publish_topic = str(self.get_parameter('publish_topic').value)     # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.use_illegal_zone_filter = self.USE_ILLEGAL_ZONE_FILTER             # ZONE구역 밖에서 검출된 객체 제외
        self.illegal_zones = self.ILLEGAL_ZONES                                 # 위에서 선언된 ZONE의 좌표값
        self.track_match_distance = max(                                        # 새 검출과 기존 객체의 맵 좌표거리 지정값 이하면 같은 객체로 판단
            0.0, float(self.get_parameter('track_match_distance').value))
        self.track_timeout = max(                                               # 객체가 사라져도 추적 정보를 유지하는 시간
            0.0, float(self.get_parameter('track_timeout').value))
        self.next_track_id = 0                                                  # 새로운 객체에 부여할 ID
        self.tracked_objects = []                                               # 현재 추적 중인 객체 목록
        self.last_published_active_ids = None                                   
        self.last_subscription_count = 0
        self._open_camera()
        self.annotated_image_publisher = self.create_publisher(                 # 검출 결과 영상 Publisher 생성
            Image, self.annotated_image_topic, 10)
        object_qos = QoSProfile(                                                # QoSProfile설정
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            String, self.publish_topic, object_qos)                             # 마지막 객체 상태를 보관하여 늦게 연결된 구독자에게도 전달합니다.
        self.processing = False								                    # 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.
        self.create_timer(                                                      # 설정된 주기마다 프레임 처리 콜백을 실행하는 타이머 생성
            float(self.get_parameter('frame_period').value),
            self.process_frames,
        )
        self.get_logger().info(                                                 # 노드 초기화 완료 정보를 로그로 출력
            f'Ready with webcam {self.camera_name}; '
            f'publishing {self.publish_topic}')			                        # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.

    def _declare_parameters(self):		# ROS 파라미터 기본값들을 선언하는 함수를 정의합니다.
        self.declare_parameter('camera_index', 0)                               # 사용할 카메라 장치 인덱스 목록
        self.declare_parameter('calibration_path', '')                          # 카메라 캘리브레이션 JSON 경로
        self.declare_parameter('model_path', '')					            # YOLO 모델 파일 경로 파라미터를 선언
        self.declare_parameter('class_ids', [])						            # 필터링할 클래스 ID 목록 파라미터를 선언
        self.declare_parameter('confidence_threshold', 0.70)				    # 객체 감지 최소 신뢰도 기본값을 선언
        self.declare_parameter('frame_period', 0.1)					            # 프레임 처리 주기 기본값을 초 단위로 선언
        self.declare_parameter('show_image', True)					            # 영상 출력 여부 기본값을 선언
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('display_height', 720)
        self.declare_parameter(
            'publish_topic', '/webcam_objects/map_detections')				    # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.declare_parameter('track_match_distance', 1)                     # 현재 검출을 기존 트랙과 연결할 최대 맵 거리 선언
        self.declare_parameter('track_timeout', 10)                              # 마지막 검출 이후 트랙을 유지할 시간 선언

    def _load_yolo_model(self):							        # YOLO 모델을 불러오는 함수를 정의합니다.
        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        if not model_path:
            raise RuntimeError('model_path is required')
        if not os.path.isfile(model_path):
            raise RuntimeError(f'YOLO model not found: {model_path}')
        from ultralytics import YOLO
        return YOLO(model_path)

    def _model_class_names(self):							# YOLO 모델의 클래스 ID와 클래스 이름을 딕셔너리로 변환
        names = getattr(self.model, 'names', {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def _open_camera(self):                                 # 웹캠과 호모그래피 설정을 불러오고 카메라 장치연결
        self.camera_index = int(self.get_parameter('camera_index').value)
        self.camera_name = self.CAMERA_NAME
        self.annotated_image_topic = self.ANNOTATED_IMAGE_TOPIC
        calibration_path = str(
            self.get_parameter('calibration_path').value)
        self.homography = self._load_homography(
            calibration_path, self.camera_name)
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(
                f'Cannot open {self.camera_name} at camera index '
                f'{self.camera_index}')
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'{self.camera_name}: index={self.camera_index}, '
            f'size={width}x{height}, '
            f'annotated_image_topic={self.annotated_image_topic}')

    def _bgr_to_image_message(self, frame, frame_id):       # OpenCV BGR 이미지를 ROS2 Image 메시지로 변환 (OpenCV이미지는 Topic으로 바로 보낼 수 없기 때문에, sensor_msgs/image형식으로 변환
        height, width, channels = frame.shape
        if channels != 3:
            raise RuntimeError('expected BGR image with 3 channels')

        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.height = height
        message.width = width
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = width * channels
        message.data = frame.tobytes()
        return message

    def _load_homography(self, path, camera_name):			# 캘리브레이션 JSON에서 카메라의 호모그래피 행렬 로드
        expanded_path = os.path.expanduser(str(path))					        
        if not expanded_path:
            raise RuntimeError(f'calibration_path is required for {camera_name}')
        with open(expanded_path, encoding='utf-8') as stream:				    # 컨텍스트 관리 블록을 시작합니다.
            calibration = json.load(stream)						                # JSON 파일 내용을 파이썬 dict로 읽습니다.

        image_points = np.asarray(							                # 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
            calibration.get('image_points', []), dtype=np.float32)
        map_points = np.asarray(							                # 캘리브레이션에 사용할 맵 기준점들을 다룹니다.
            calibration.get('map_points', []), dtype=np.float32)
        if image_points.shape != map_points.shape:
            raise RuntimeError(
                f'{camera_name}: image_points and map_points shape mismatch')
        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise RuntimeError(
                f'{camera_name}: calibration points must be [x, y] pairs')
        if len(image_points) < 4:
            raise RuntimeError(
                f'{camera_name}: at least four calibration pairs are required')

        homography, mask = cv2.findHomography(						# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
            image_points, map_points, cv2.RANSAC)					# 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
        if homography is None:								# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
            raise RuntimeError(f'{camera_name}: failed to calculate homography')	
        inliers = int(mask.sum()) if mask is not None else len(image_points)		# 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
        self.get_logger().info(
            f'{camera_name}: loaded {len(image_points)} calibration pairs '
            f'({inliers} inliers)')
        return homography								# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 반환

    def process_frames(self):					# 카메라 프레임을 읽어 객체 검출·좌표 변환·추적·발행 수행
        if self.processing:
            return
        self.processing = True
        try:
            success, frame = self.capture.read()
            if not success:
                self.get_logger().warn(
                    f'{self.camera_name}: failed to read frame',
                    throttle_duration_sec=2.0)
                return
            detections = self._detect_objects(frame)

            system_time = time.time()
            trackable_detections = [
                detection for detection in detections
                if self._is_json_publish_candidate(detection)
            ]
            tracked_objects = self._update_tracks(
                trackable_detections, system_time)
            self._draw_illegal_zones(frame)
            self._draw_detections(frame, detections)
            self.annotated_image_publisher.publish(
                self._bgr_to_image_message(frame, self.camera_name))
            if self.show_image:
                self._show_frame(frame)
            payload = {
                'system_time': system_time,
                'objects': tracked_objects,
            }
            if self._should_publish_objects(tracked_objects):
                self._publish_payload(payload)
            if self.show_image:
                cv2.waitKey(1)								# OpenCV 창 갱신과 키 입력 처리를 수행합니다.
        except Exception as error:
            self.get_logger().error(f'Frame processing failed: {error}')
        finally:									        # 성공 여부와 관계없이 항상 실행할 정리 블록입니다.
            self.processing = False							# 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.

    def _show_frame(self, frame):
        if not self.window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.window_name, self.display_width, self.display_height)
            self.window_initialized = True
        cv2.imshow(self.window_name, frame)

    def _detect_objects(self, frame):                       # YOLO로 객체를 검출하고 클래스·신뢰도 조건을 만족하는 결과만 반환
        detections = []
        for result in self.model(frame, verbose=False):
            for box in result.boxes:
                class_id = int(box.cls[0])
                if self.class_ids and class_id not in self.class_ids:
                    continue
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:				# 객체 감지 결과를 사용할 최소 신뢰도를 다룹니다.
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                detection = {
                    'camera': self.camera_name,
                    'camera_index': self.camera_index,
                    'class_id': class_id,
                    'class_name': self.class_names.get(
                        class_id, f'class_{class_id}'),
                    'confidence': confidence,
                    'bbox_xyxy': [x1, y1, x2, y2],
                }

                pixel_x = (x1 + x2) / 2.0
                pixel_y = y2
                map_x, map_y = self._pixel_to_map(
                    self.homography, pixel_x, pixel_y)
                detection['pixel'] = {'x': pixel_x, 'y': pixel_y}
                detection['map'] = {'x': map_x, 'y': map_y}
                detection['in_illegal_zone'] = self._is_in_illegal_zone(
                    map_x, map_y)
                detections.append(detection)
        return detections

    def _pixel_to_map(self, homography, pixel_x, pixel_y):		# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
        pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        map_point = cv2.perspectiveTransform(pixel, homography)
        return float(map_point[0, 0, 0]), float(map_point[0, 0, 1])

    def _is_in_illegal_zone(self, map_x, map_y):                # 맵 좌표가 불법주차 감시 구역 중 하나에 포함되는지 확인
        if not self.use_illegal_zone_filter:
            return True
        for zone in self.illegal_zones:
            if self._is_in_zone(zone, map_x, map_y):
                return True
        return False

    def _find_zone_id(self, map_x, map_y):                      # 맵 좌표가 포함된 감시 구역 번호를 반환
        for zone_id, zone in enumerate(self.illegal_zones, start=1):
            if self._is_in_zone(zone, map_x, map_y):
                return zone_id
        return None

    def _is_in_zone(self, zone, map_x, map_y):                  # 하나의 사각형 또는 다각형 구역 내부에 좌표가 있는지 판정
        zone_type = zone.get('type', 'box')
        if zone_type == 'polygon':
            points = np.asarray(zone.get('points', []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                return False
            return cv2.pointPolygonTest(points, (map_x, map_y), False) >= 0

        min_x = min(zone['min_x'], zone['max_x'])
        max_x = max(zone['min_x'], zone['max_x'])
        min_y = min(zone['min_y'], zone['max_y'])
        max_y = max(zone['min_y'], zone['max_y'])
        return min_x <= map_x <= max_x and min_y <= map_y <= max_y

    def _number_by_confidence(self, objects):                   # 검출 객체를 신뢰도순으로 정렬하고 임시 ID와 맵 좌표 부여
        numbered_objects = []
        for object_id, object_data in enumerate(sorted(
                objects, key=lambda item: item['confidence'], reverse=True)):
            numbered_objects.append({
                'id': object_id,
                'x': object_data['map']['x'],
                'y': object_data['map']['y'],
                'z': 0.0,
            })
        return numbered_objects

    def _should_publish_objects(self, objects):                 # 객체 상태가 변경됐거나 새 구독자가 연결됐을때만 발행하도록 판정
        subscription_count = self.publisher.get_subscription_count()
        subscriber_joined = subscription_count > self.last_subscription_count
        self.last_subscription_count = subscription_count

        active_states = tuple(sorted(
            (object_data['event_id'], object_data['zone'])
            for object_data in objects
        ))
        active_changed = active_states != self.last_published_active_ids
        if not subscriber_joined and not active_changed:
            return False
        self.last_published_active_ids = active_states
        return True

    def _draw_illegal_zones(self, frame):                       # 맵의 감시 구역을 영상 좌표로 변환해 카메라 화면에 표시
        if not self.use_illegal_zone_filter:
            return
        inverse_homography = np.linalg.inv(self.homography)
        overlay = frame.copy()
        zone_pixel_points = []
        for zone_index, zone in enumerate(self.illegal_zones, start=1):
            pixel_points = self._zone_to_pixel_points(zone, inverse_homography)
            if pixel_points is None:
                continue
            cv2.fillPoly(overlay, [pixel_points], (0, 0, 255))
            zone_pixel_points.append((zone_index, pixel_points))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        for zone_index, pixel_points in zone_pixel_points:
            cv2.polylines(frame, [pixel_points], True, (0, 0, 255), 2)
            label_x = int(np.mean(pixel_points[:, 0]))
            label_y = int(np.mean(pixel_points[:, 1]))
            cv2.putText(
                frame,
                f'Zone {zone_index}',
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    def _zone_to_pixel_points(self, zone, inverse_homography):  # 맵 좌표로 정의된 감시 구역 꼭짓점을 영상 픽셀 좌표로 변환
        zone_type = zone.get('type', 'box')
        if zone_type == 'polygon':
            map_points = np.asarray(zone.get('points', []), dtype=np.float32)
        else:
            min_x = min(zone['min_x'], zone['max_x'])
            max_x = max(zone['min_x'], zone['max_x'])
            min_y = min(zone['min_y'], zone['max_y'])
            max_y = max(zone['min_y'], zone['max_y'])
            map_points = np.asarray([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ], dtype=np.float32)

        if map_points.ndim != 2 or map_points.shape[0] < 3 or map_points.shape[1] != 2:
            return None
        points = np.array([map_points], dtype=np.float32)
        pixel_points = cv2.perspectiveTransform(points, inverse_homography)
        return np.rint(pixel_points[0]).astype(np.int32)

    def _update_tracks(self, objects, system_time):          # 현재 검출 결과를 기존 트랙과 연결하고 객체 추적 상태 갱신
        used_track_ids = set()
        for object_data in sorted(
                objects, key=lambda item: item['confidence'], reverse=True):
            zone_id = self._find_zone_id(
                object_data['map']['x'],
                object_data['map']['y'],
            )
            bbox = self._best_bbox_size(object_data)
            track = self._find_matching_track(object_data, used_track_ids)
            if track is None:
                track = {
                    'id': self.next_track_id,
                    'class_id': object_data['class_id'],
                    'zone': zone_id,
                    'x': object_data['map']['x'],
                    'y': object_data['map']['y'],
                    'bbox': bbox,
                    'last_seen': system_time,
                }
                self.next_track_id += 1
                self.tracked_objects.append(track)
            else:
                track['zone'] = zone_id
                track['x'] = object_data['map']['x']
                track['y'] = object_data['map']['y']
                track['bbox'] = bbox
                track['last_seen'] = system_time

            used_track_ids.add(track['id'])
            object_data['track_id'] = track['id']

        self.tracked_objects = [
            track for track in self.tracked_objects
            if system_time - track['last_seen'] <= self.track_timeout
        ]
        tracked_payload = [
            {
                'event_id': track['id'],
                'zone': track['zone'],
                'x': track['x'],
                'y': track['y'],
                'z': 0.0,
                'bbox': dict(track['bbox']),
            }
            for track in sorted(
                self.tracked_objects, key=lambda item: item['id'])
        ]
        return tracked_payload

    def _is_fire_truck(self, object_data):
        """Return True for class-4 fire trucks."""
        return object_data['class_id'] == self.FIRE_TRUCK_CLASS_ID

    def _is_json_publish_candidate(self, object_data):
        """Return True only for car detections that should be tracked/published."""
        return (
            object_data['class_id'] == self.JSON_CLASS_ID
            and object_data.get('in_illegal_zone', False)
            and 'map' in object_data
        )

    @staticmethod
    def _best_bbox_size(object_data):       # 여러 카메라 검출 정보 중 가장 큰 바운딩 박스 크기 반환
        bbox_xyxy = object_data.get('bbox_xyxy')
        if bbox_xyxy is None:
            return {'width': 0.0, 'height': 0.0}

        x1, y1, x2, y2 = bbox_xyxy
        return {
            'width': max(0.0, float(x2) - float(x1)),
            'height': max(0.0, float(y2) - float(y1)),
        }

    def _find_matching_track(self, object_data, used_track_ids):    # 거리·구역·바운딩 박스 조건을 이용해 가장 적합한 기존 트랙 검색
        if self.track_match_distance <= 0.0:
            return None
        best_track = None
        best_distance = self.track_match_distance
        object_x = object_data['map']['x']
        object_y = object_data['map']['y']
        for track in self.tracked_objects:
            if track['id'] in used_track_ids:
                continue
            if track['class_id'] != object_data['class_id']:
                continue
            dx = track['x'] - object_x
            dy = track['y'] - object_y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= best_distance:
                best_track = track
                best_distance = distance
        return best_track

    def _draw_detections(self, frame, detections):		# 객체 바운딩 박스, 클래스명, 신뢰도, 추적 ID를 영상에 표시
        for detection in detections:
            if not detection.get('in_illegal_zone', False):
                continue

            x1, y1, x2, y2 = [
                int(value) for value in detection['bbox_xyxy']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            map_x = detection['map']['x']
            map_y = detection['map']['y']
            track_id = detection.get('track_id', '?')
            label = (
                f"id={track_id} "
                f"{detection['class_name']} {detection['confidence']:.2f} "
                f"map=({map_x:.2f},{map_y:.2f})")
            cv2.circle(
                frame,
                (int(detection['pixel']['x']), int(detection['pixel']['y'])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                label,	
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def _publish_payload(self, payload):	    # 객체 좌표 데이터를 JSON 문자열로 변환해 ROS2토픽으로 발행
        message = String()
        message.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(message)

    def _parse_int_list(self, value):			# 문자열이나 배열 형태의 클래스 ID 파라미터를 정수 리스트로 변환
        if isinstance(value, str):							
            if not value.strip():							
                return []								
            return [int(item.strip()) for item in value.split(',')]			
        return [int(item) for item in value]						

    def destroy_node(self):						# 카메라와 OpenCV 창을 정리한 후 ROS2 노드 종료
        capture = getattr(self, 'capture', None)
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()								
        super().destroy_node()								


def main(args=None):
    rclpy.init(args=args)								# ROS2 파이썬 클라이언트를 초기화합니다.
    node = None										    # 노드 변수를 먼저 None으로 초기화합니다.
    try:										
        node = MultiWebcamObjectMapper5()						
        rclpy.spin(node)								# ROS2 노드의 콜백이 계속 실행되게 합니다.
    except KeyboardInterrupt:							# Ctrl+C로 종료하는 경우를 처리합니다.
        pass
    except Exception as error:
        if node is not None:								
            node.get_logger().fatal(str(error))						
        else:										
            print(f'Failed to start multi_webcam_object_mapper5: {error}')		
    finally:
        if node is not None:								
            node.destroy_node()								
        if rclpy.ok():									
            rclpy.shutdown()								# ROS2 클라이언트를 종료합니다.


if __name__ == '__main__':
    main()
