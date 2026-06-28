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


class MultiWebcamObjectMapper(Node):
    """Detect objects from one fixed webcam and publish map coordinates."""

    NODE_NAME = 'multi_webcam_object_mapper4'
    WINDOW_PREFIX = 'multi_webcam_object_mapper4'
    CAMERA_NAME = 'webcam1'
    ANNOTATED_IMAGE_TOPIC = '/webcam_images/webcam1/detections'
    USE_ILLEGAL_ZONE_FILTER = True
    SLOT_WIDTH = 0.35
    SLOT_HEIGHT = 0.35
    DRAW_SLOTS = False
    ILLEGAL_ZONES = [
        {
            'type': 'polygon',
            'single_slot': True,
            'points': [
                [-3.3, -0.23],
                [-3.25, 0.05],
                [-1.38, 0.03],
                [-1.47, -0.319],
            ],
        },
        {
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
            'type': 'polygon',
            'single_slot': True,
            'points': [
                [-3.85, 0.95],
                [-3.81, 1.57],
                [-3.1, 1.57],
                [-3.15, 0.95],
            ],
        },
        {
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
        super().__init__(self.NODE_NAME)					                    # ROS 노드 이름
        self._declare_parameters()							                # 노드에서 사용할 ROS 파라미터선언

        self.model = self._load_yolo_model()						        # YOLO 객체 감지 모델을 불러옵니다.
        self.class_names = self._model_class_names()					    # 모델의 클래스 ID와 이름 목록을 가져옵니다.
        self.confidence_threshold = float(						            # 객체 감지 결과를 사용할 최소 신뢰도를 다룹니다.
            self.get_parameter('confidence_threshold').value)
        self.class_ids = self._parse_int_list(						            # 감지할 클래스 ID 필터 목록을 다룹니다.
            self.get_parameter('class_ids').value)
        self.show_image = bool(self.get_parameter('show_image').value)			# 카메라 영상 창 표시 여부를 다룹니다.
        self.publish_topic = str(self.get_parameter('publish_topic').value)		# 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.use_illegal_zone_filter = self.USE_ILLEGAL_ZONE_FILTER
        self.illegal_zones = self.ILLEGAL_ZONES
        self.parking_slots = self._build_parking_slots()
        self.track_match_distance = max(
            0.0, float(self.get_parameter('track_match_distance').value))
        self.track_timeout = max(
            0.0, float(self.get_parameter('track_timeout').value))
        self.next_track_id = 0
        self.tracked_objects = []
        self.last_published_active_ids = None
        self.previous_detected_ids = set()
        self.last_subscription_count = 0

        self._open_camera()
        self.annotated_image_publisher = self.create_publisher(
            Image, self.annotated_image_topic, 10)
        object_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            String, self.publish_topic, object_qos)                              # 마지막 객체 상태를 보관하여 늦게 연결된 구독자에게도 전달합니다.
        self.processing = False								                    # 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.
        self.create_timer(
            float(self.get_parameter('frame_period').value),
            self.process_frames,
        )
        self.get_logger().info(
            f'Ready with webcam {self.camera_name}; '
            f'{len(self.parking_slots)} slots; '
            f'publishing {self.publish_topic}')			                        # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.

    def _declare_parameters(self):		# ROS 파라미터 기본값들을 선언하는 함수를 정의합니다.
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('model_path', '')					            # YOLO 모델 파일 경로 파라미터를 선언합니다.
        self.declare_parameter('class_ids', [])						            # 필터링할 클래스 ID 목록 파라미터를 선언합니다.
        self.declare_parameter('confidence_threshold', 0.70)				    # 객체 감지 최소 신뢰도 기본값을 선언합니다.
        self.declare_parameter('frame_period', 0.1)					            # 프레임 처리 주기 기본값을 초 단위로 선언합니다.
        self.declare_parameter('show_image', True)					            # 영상 출력 여부 기본값을 선언합니다.
        self.declare_parameter(
            'publish_topic', '/webcam_objects/map_detections')				    # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.declare_parameter('track_match_distance', 0.6)
        self.declare_parameter('track_timeout', 1.5)

    def _load_yolo_model(self):							        # YOLO 모델을 불러오는 함수를 정의합니다.
        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        if not model_path:
            raise RuntimeError('model_path is required')
        if not os.path.isfile(model_path):
            raise RuntimeError(f'YOLO model not found: {model_path}')
        from ultralytics import YOLO
        return YOLO(model_path)

    def _model_class_names(self):							# 모델 클래스 이름 목록을 정리하는 함수를 정의합니다.
        names = getattr(self.model, 'names', {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def _open_camera(self):
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

    def _bgr_to_image_message(self, frame, frame_id):                           # OpenCV BGR 이미지를 ROS2 Image 메시지로 변환 (OpenCV이미지는 Topic으로 바로 보낼 수 없기 때문에, sensor_msgs/image형식으로 변환
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

    def _load_homography(self, path, camera_name):					            # 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
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

    def process_frames(self):								# 타이머마다 카메라 프레임을 처리하는 (핵심)함수
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
            tracked_objects, detected_ids = self._update_tracks(
                detections, system_time)
            self._draw_illegal_zones(frame)
            self._draw_detections(frame, detections)
            self.annotated_image_publisher.publish(
                self._bgr_to_image_message(frame, self.camera_name))
            if self.show_image:
                cv2.imshow(
                    f'{self.WINDOW_PREFIX} {self.camera_name}', frame)
            payload = {
                'system_time': system_time,
                'objects': tracked_objects,
            }
            if self._should_publish_objects(tracked_objects, detected_ids):
                self._publish_payload(payload)
            if self.show_image:
                cv2.waitKey(1)								# OpenCV 창 갱신과 키 입력 처리를 수행합니다.
        except Exception as error:
            self.get_logger().error(f'Frame processing failed: {error}')
        finally:									        # 성공 여부와 관계없이 항상 실행할 정리 블록입니다.
            self.processing = False							# 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.

    def _detect_objects(self, frame):
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
                pixel_x = (x1 + x2) / 2.0
                pixel_y = y2
                map_x, map_y = self._pixel_to_map(
                    self.homography, pixel_x, pixel_y)
                if not self._is_in_illegal_zone(map_x, map_y):
                    continue
                detections.append({
                    'camera': self.camera_name,
                    'camera_index': self.camera_index,
                    'class_id': class_id,
                    'class_name': self.class_names.get(
                        class_id, f'class_{class_id}'),
                    'confidence': confidence,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'pixel': {'x': pixel_x, 'y': pixel_y},
                    'map': {'x': map_x, 'y': map_y},
                })
        return detections

    def _pixel_to_map(self, homography, pixel_x, pixel_y):				# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
        pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        map_point = cv2.perspectiveTransform(pixel, homography)
        return float(map_point[0, 0, 0]), float(map_point[0, 0, 1])

    def _is_in_illegal_zone(self, map_x, map_y):
        if not self.use_illegal_zone_filter:
            return True
        for zone in self.illegal_zones:
            if self._is_in_zone(zone, map_x, map_y):
                return True
        return False

    def _find_zone_id(self, map_x, map_y):
        for zone_id, zone in enumerate(self.illegal_zones, start=1):
            if self._is_in_zone(zone, map_x, map_y):
                return zone_id
        return None

    def _is_in_zone(self, zone, map_x, map_y):
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

    def _build_parking_slots(self):
        slots = []
        slot_id = 0
        for zone_index, zone in enumerate(self.illegal_zones, start=1):
            zone_points = self._zone_map_points(zone)
            if zone_points is None:
                continue
            if zone.get('single_slot', False):
                slots.append({
                    'id': slot_id,
                    'zone': zone_index,
                    'points': zone_points.tolist(),
                })
                slot_id += 1
                continue
            min_x = float(np.min(zone_points[:, 0]))
            max_x = float(np.max(zone_points[:, 0]))
            min_y = float(np.min(zone_points[:, 1]))
            max_y = float(np.max(zone_points[:, 1]))
            x = max_x - self.SLOT_WIDTH
            while x >= min_x:
                y = min_y
                while y < max_y:
                    center_x = x + self.SLOT_WIDTH / 2.0
                    center_y = y + self.SLOT_HEIGHT / 2.0
                    if self._is_in_zone(zone, center_x, center_y):
                        slots.append({
                            'id': slot_id,
                            'zone': zone_index,
                            'points': [
                                [x, y],
                                [x + self.SLOT_WIDTH, y],
                                [x + self.SLOT_WIDTH, y + self.SLOT_HEIGHT],
                                [x, y + self.SLOT_HEIGHT],
                            ],
                        })
                        slot_id += 1
                    y += self.SLOT_HEIGHT
                x -= self.SLOT_WIDTH
        return slots

    def _zone_map_points(self, zone):
        zone_type = zone.get('type', 'box')
        if zone_type == 'polygon':
            points = np.asarray(zone.get('points', []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                return None
            return points

        min_x = min(zone['min_x'], zone['max_x'])
        max_x = max(zone['min_x'], zone['max_x'])
        min_y = min(zone['min_y'], zone['max_y'])
        max_y = max(zone['min_y'], zone['max_y'])
        return np.asarray([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ], dtype=np.float32)

    def _assign_slots(self, objects):
        slot_objects = {}
        for object_data in sorted(
                objects, key=lambda item: item['confidence'], reverse=True):
            map_x = object_data['map']['x']
            map_y = object_data['map']['y']
            slot = self._find_slot(map_x, map_y)
            if slot is None or slot['id'] in slot_objects:
                continue
            slot_objects[slot['id']] = {
                'id': slot['id'],
                'x': map_x,
                'y': map_y,
                'z': 0.0,
            }
        return [
            slot_objects[slot_id]
            for slot_id in sorted(slot_objects)
        ]

    def _find_slot(self, map_x, map_y):
        for slot in self.parking_slots:
            points = np.asarray(slot['points'], dtype=np.float32)
            if cv2.pointPolygonTest(points, (map_x, map_y), False) >= 0:
                return slot
        return None

    def _number_by_confidence(self, objects):
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

    def _should_publish_objects(self, objects, detected_ids):
        subscription_count = self.publisher.get_subscription_count()
        subscriber_joined = subscription_count > self.last_subscription_count
        self.last_subscription_count = subscription_count

        active_states = tuple(sorted(
            (object_data['event_id'], object_data['zone'])
            for object_data in objects
        ))
        appeared = bool(detected_ids - self.previous_detected_ids)
        active_changed = active_states != self.last_published_active_ids
        self.previous_detected_ids = set(detected_ids)
        if not subscriber_joined and not appeared and not active_changed:
            return False
        self.last_published_active_ids = active_states
        return True

    def _draw_illegal_zones(self, frame):
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

        if self.DRAW_SLOTS:
            self._draw_slots(frame, inverse_homography)

    def _zone_to_pixel_points(self, zone, inverse_homography):
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

    def _draw_slots(self, frame, inverse_homography):
        for slot in self.parking_slots:
            pixel_points = self._map_points_to_pixel_points(
                slot['points'], inverse_homography)
            if pixel_points is None:
                continue
            cv2.polylines(frame, [pixel_points], True, (0, 255, 255), 1)
            label_x = int(np.mean(pixel_points[:, 0]))
            label_y = int(np.mean(pixel_points[:, 1]))
            cv2.putText(
                frame,
                str(slot['id']),
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )

    def _map_points_to_pixel_points(self, map_points, inverse_homography):
        map_points = np.asarray(map_points, dtype=np.float32)
        if map_points.ndim != 2 or map_points.shape[0] < 3 or map_points.shape[1] != 2:
            return None
        points = np.array([map_points], dtype=np.float32)
        pixel_points = cv2.perspectiveTransform(points, inverse_homography)
        return np.rint(pixel_points[0]).astype(np.int32)

    def _update_tracks(self, objects, system_time):
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
        return tracked_payload, set(used_track_ids)

    @staticmethod
    def _best_bbox_size(object_data):
        bbox_xyxy = object_data.get('bbox_xyxy')
        if bbox_xyxy is None:
            return {'width': 0.0, 'height': 0.0}

        x1, y1, x2, y2 = bbox_xyxy
        return {
            'width': max(0.0, float(x2) - float(x1)),
            'height': max(0.0, float(y2) - float(y1)),
        }

    def _find_matching_track(self, object_data, used_track_ids):
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

    def _draw_detections(self, frame, detections):					# 감지 결과를 영상 프레임 위에 그리는 함수를 정의합니다.
        for detection in detections:
            x1, y1, x2, y2 = [
                int(value) for value in detection['bbox_xyxy']]
            map_x = detection['map']['x']
            map_y = detection['map']['y']
            track_id = detection.get('track_id', '?')
            label = (
                f"id={track_id} "
                f"{detection['class_name']} {detection['confidence']:.2f} "
                f"map=({map_x:.2f},{map_y:.2f})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

    def _publish_payload(self, payload):						# 감지 결과 payload를 publish하는 함수를 정의합니다.
        message = String()
        message.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(message)

    def _parse_int_list(self, value):						# 파라미터 값을 정수 리스트로 변환하는 함수를 정의합니다.
        if isinstance(value, str):							
            if not value.strip():							
                return []								
            return [int(item.strip()) for item in value.split(',')]			
        return [int(item) for item in value]						

    def destroy_node(self):								# 노드 종료 시 리소스를 정리하는 함수를 정의합니다.
        capture = getattr(self, 'capture', None)
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()								
        super().destroy_node()								


def main(args=None):
    rclpy.init(args=args)								# ROS2 파이썬 클라이언트를 초기화합니다.
    node = None										    # 노드 변수를 먼저 None으로 초기화합니다.
    try:										
        node = MultiWebcamObjectMapper()						
        rclpy.spin(node)								# ROS2 노드의 콜백이 계속 실행되게 합니다.
    except KeyboardInterrupt:							# Ctrl+C로 종료하는 경우를 처리합니다.
        pass
    except Exception as error:
        if node is not None:								
            node.get_logger().fatal(str(error))						
        else:										
            print(f'Failed to start multi_webcam_object_mapper4: {error}')		
    finally:
        if node is not None:								
            node.destroy_node()								
        if rclpy.ok():									
            rclpy.shutdown()								# ROS2 클라이언트를 종료합니다.


if __name__ == '__main__':
    main()
