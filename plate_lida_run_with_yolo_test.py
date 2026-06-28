# plate_lida_run_with_yolo_test.py
#
# 목적:
#   1. YOLO 화면을 켠 상태로 번호판 탐지 상태를 확인합니다.
#   2. 번호판이 안 보이면 라이다 직선으로 차량 옆면과 평행하게 맞춥니다.
#   3. 초록 선 길이 기준으로 짧게 전진/후진합니다.
#   4. 입력한 차량 map 좌표를 바라본 뒤 YOLO를 몇 초간 확인합니다.
#
# 주의:
#   이동 중 YOLO 탐지는 기본적으로 "화면 표시용"입니다.
#   다른 차량 번호판을 잘못 성공 처리하지 않도록, 성공 판정은 정지 후 확인 단계에서만 합니다.

import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from ultralytics import YOLO
from visualization_msgs.msg import Marker


class PlateLidarRunWithYoloTestNode(Node):
    """라이다 보정 이동 후 목표 좌표를 바라보고 YOLO로 번호판을 확인하는 테스트 노드."""

    def __init__(self):
        super().__init__('plate_lidar_run_with_yolo_test_node')

        # robot4 기본 토픽입니다.
        self.declare_parameter('scan_topic', '/robot4/scan')
        self.declare_parameter('odom_topic', '/robot4/odom')
        self.declare_parameter('amcl_pose_topic', '/robot4/amcl_pose')
        self.declare_parameter('cmd_vel_topic', '/robot4/cmd_vel')
        self.declare_parameter('camera_topic', '/robot4/oakd/rgb/image_raw/compressed')

        # YOLO 설정입니다.
        self.declare_parameter(
            'model_path',
            '/home/rokey/rokey_ws/src/final_pjt/final_pjt/semi_allimages_v5n.pt'
        )
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('plate_class_id', 2)
        self.declare_parameter('show_yolo_window', True)
        self.declare_parameter('yolo_min_period_sec', 0.20)
        self.declare_parameter('initial_yolo_check_sec', 1.5)
        self.declare_parameter('final_yolo_check_sec', 3.0)

        # target_x/y는 map 좌표입니다.
        # use_target_pose=True로 실행하면 이동 후 해당 좌표 방향을 바라봅니다.
        self.declare_parameter('use_target_pose', False)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)

        # robot4에서 OAK-D 카메라 정면은 /scan 각도 기준 대략 -90도였습니다.
        self.declare_parameter('camera_forward_angle_deg', -90.0)

        # 카메라 정면 기준 라이다 ROI입니다.
        self.declare_parameter('roi_angle_min_deg', -125.0)
        self.declare_parameter('roi_angle_max_deg', -55.0)
        self.declare_parameter('roi_range_min_m', 0.15)
        self.declare_parameter('roi_range_max_m', 0.70)

        # 라이다 직선 추출 파라미터입니다.
        self.declare_parameter('min_points', 12)
        self.declare_parameter('cluster_distance_threshold_m', 0.12)
        self.declare_parameter('ransac_iterations', 80)
        self.declare_parameter('ransac_distance_threshold_m', 0.035)
        self.declare_parameter('min_line_length_m', 0.15)

        # cmd_vel + odom 이동 파라미터입니다.
        self.declare_parameter('rotation_sign', 1.0)
        self.declare_parameter('angular_speed_rad_s', 0.25)
        self.declare_parameter('linear_speed_m_s', 0.05)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.declare_parameter('distance_tolerance_m', 0.015)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter('move_direction', -1.0)  # +1.0 전진, -1.0 후진
        self.declare_parameter('line_length_multiplier', 1.0)
        self.declare_parameter('max_search_distance_m', 1.50)

        # 전진 중 안전 정지입니다. 후진 안전 검사는 아직 넣지 않았습니다.
        self.declare_parameter('enable_front_safety_stop', True)
        self.declare_parameter('front_safety_min_deg', -125.0)
        self.declare_parameter('front_safety_max_deg', -55.0)
        self.declare_parameter('front_safety_distance_m', 0.25)

        self.declare_parameter('wait_timeout_sec', 10.0)
        self.declare_parameter('log_period_sec', 0.5)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.amcl_pose_topic = self.get_parameter('amcl_pose_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value

        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.plate_class_id = self.get_parameter('plate_class_id').value
        self.show_yolo_window = self.get_parameter('show_yolo_window').value
        self.yolo_min_period = self.get_parameter('yolo_min_period_sec').value
        self.initial_yolo_check_sec = self.get_parameter('initial_yolo_check_sec').value
        self.final_yolo_check_sec = self.get_parameter('final_yolo_check_sec').value

        self.use_target_pose = self.get_parameter('use_target_pose').value
        self.target_x = self.get_parameter('target_x').value
        self.target_y = self.get_parameter('target_y').value

        self.camera_forward_angle = math.radians(
            self.get_parameter('camera_forward_angle_deg').value
        )
        self.roi_angle_min = math.radians(self.get_parameter('roi_angle_min_deg').value)
        self.roi_angle_max = math.radians(self.get_parameter('roi_angle_max_deg').value)
        self.roi_range_min = self.get_parameter('roi_range_min_m').value
        self.roi_range_max = self.get_parameter('roi_range_max_m').value
        self.min_points = self.get_parameter('min_points').value
        self.cluster_distance_threshold = self.get_parameter(
            'cluster_distance_threshold_m'
        ).value
        self.ransac_iterations = self.get_parameter('ransac_iterations').value
        self.ransac_distance_threshold = self.get_parameter(
            'ransac_distance_threshold_m'
        ).value
        self.min_line_length = self.get_parameter('min_line_length_m').value
        self.rotation_sign = self.get_parameter('rotation_sign').value
        self.angular_speed = self.get_parameter('angular_speed_rad_s').value
        self.linear_speed = self.get_parameter('linear_speed_m_s').value
        self.angle_tolerance = math.radians(self.get_parameter('angle_tolerance_deg').value)
        self.distance_tolerance = self.get_parameter('distance_tolerance_m').value
        self.settle_time = self.get_parameter('settle_time_sec').value
        self.move_direction = self.get_parameter('move_direction').value
        self.line_length_multiplier = self.get_parameter('line_length_multiplier').value
        self.max_search_distance = self.get_parameter('max_search_distance_m').value
        self.enable_front_safety_stop = self.get_parameter(
            'enable_front_safety_stop'
        ).value
        self.front_safety_min = math.radians(self.get_parameter('front_safety_min_deg').value)
        self.front_safety_max = math.radians(self.get_parameter('front_safety_max_deg').value)
        self.front_safety_distance = self.get_parameter('front_safety_distance_m').value
        self.wait_timeout = self.get_parameter('wait_timeout_sec').value
        self.log_period_sec = self.get_parameter('log_period_sec').value

        model_path = self.get_parameter('model_path').value
        self.model = self._load_yolo_model(model_path)
        self.class_names = self.model.names
        self.bridge = CvBridge()
        self.window_name = 'plate_lidar_yolo_view'

        self.last_log_time = 0.0
        self.latest_scan = None
        self.latest_line = None
        self.current_odom = None
        self.current_yaw = None
        self.current_map_x = None
        self.current_map_y = None
        self.current_map_yaw = None

        self.last_yolo_time = 0.0
        self.latest_plate_detected = False
        self.latest_plate_stamp = 0.0
        self.latest_plate_confidence = 0.0
        self.accept_plate_detection = False
        self.accepted_plate_detected = False

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.amcl_pose_callback,
            10
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(Marker, '/plate_lidar_line_marker', 10)

        self.get_logger().info(
            f"라이다+YOLO 주행 테스트 시작 "
            f"(scan: {self.scan_topic}, odom: {self.odom_topic}, "
            f"amcl: {self.amcl_pose_topic}, camera: {self.camera_topic}, "
            f"cmd_vel: {self.cmd_vel_topic})"
        )
        self.get_logger().info(f"YOLO class names: {self.class_names}")

    def run_sequence(self):
        """YOLO 초기 확인 -> 라이다 보정 -> 목표 좌표 바라보기 -> YOLO 최종 확인."""
        if not self._wait_for_ready():
            self._stop_robot()
            return False

        if self.initial_yolo_check_sec > 0.0:
            if self._wait_for_plate_detection(
                self.initial_yolo_check_sec,
                '현재 자세에서 번호판 확인'
            ):
                self.get_logger().info("초기 자세에서 번호판을 탐지했습니다.")
                return True

        line = dict(self.latest_line)
        if line['line_length'] < self.min_line_length:
            self.get_logger().error(
                f"추출된 직선이 너무 짧습니다: {line['line_length']:.2f} m"
            )
            self._stop_robot()
            return False

        rotate_angle = self.rotation_sign * line['heading_error']
        self.get_logger().info(
            f"1단계: 차량면과 평행 정렬 "
            f"(line_angle={math.degrees(line['line_angle']):.1f} deg, "
            f"heading_error={math.degrees(line['heading_error']):.1f} deg, "
            f"rotate_cmd={math.degrees(rotate_angle):.1f} deg)"
        )
        if not self._rotate_relative(rotate_angle):
            self._stop_robot()
            return False

        self._sleep_with_spin(self.settle_time)

        target_distance = min(
            line['line_length'] * self.line_length_multiplier,
            self.max_search_distance
        )
        target_offset = self._move_direction_sign() * target_distance

        self.get_logger().info(
            f"2단계: 평행 방향 이동 "
            f"(line_length={line['line_length']:.2f} m, "
            f"multiplier={self.line_length_multiplier:.2f}, "
            f"target_offset={target_offset:+.2f} m)"
        )
        if abs(target_offset) > self.distance_tolerance:
            if not self._move_straight(target_offset):
                self._stop_robot()
                return False

        self._stop_robot()
        self._sleep_with_spin(self.settle_time)

        if self.use_target_pose:
            self.get_logger().info(
                f"3단계: 차량 좌표 방향 바라보기 "
                f"(target_x={self.target_x:.3f}, target_y={self.target_y:.3f})"
            )
            if not self._face_target_pose():
                self._stop_robot()
                return False
            self._sleep_with_spin(self.settle_time)
        else:
            self.get_logger().warning(
                "use_target_pose=False 입니다. 차량 좌표 방향 회전 없이 최종 YOLO 확인을 진행합니다."
            )

        if self.final_yolo_check_sec > 0.0:
            if self._wait_for_plate_detection(
                self.final_yolo_check_sec,
                '최종 자세에서 번호판 확인'
            ):
                self.get_logger().info("최종 자세에서 번호판 탐지 성공.")
                return True

        self._stop_robot()
        self.get_logger().warning("번호판을 탐지하지 못했습니다.")
        return False

    def image_callback(self, msg):
        """카메라 compressed image를 받아 YOLO를 돌리고 화면을 표시합니다."""
        now = time.monotonic()
        if now - self.last_yolo_time < self.yolo_min_period:
            return
        self.last_yolo_time = now

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"compressed image 변환 실패: {exc}")
            return

        plate_detected, best_confidence, annotated = self._detect_plate(img)
        self.latest_plate_detected = plate_detected
        if plate_detected:
            self.latest_plate_stamp = now
            self.latest_plate_confidence = best_confidence

            if self.accept_plate_detection:
                self.accepted_plate_detected = True
                self._stop_robot()

        if self.show_yolo_window:
            cv2.imshow(self.window_name, annotated)
            cv2.waitKey(1)

    def scan_callback(self, msg):
        """LaserScan을 받을 때마다 ROI 점, cluster, 직선을 새로 계산합니다."""
        self.latest_scan = msg
        points = self._scan_to_roi_points(msg)

        if len(points) < self.min_points:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_periodically(f"ROI points 부족: {len(points)}개")
            return

        clusters = self._cluster_points(points)
        selected_cluster = self._select_nearest_cluster(clusters)

        if selected_cluster is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_periodically(
                f"선택 가능한 cluster 없음: roi_points={len(points)}, clusters={len(clusters)}"
            )
            return

        line = self._fit_line_ransac(selected_cluster)
        if line is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_periodically("직선을 안정적으로 찾지 못했습니다.")
            return

        point_on_line, direction, inliers = line
        line_angle = math.atan2(direction[1], direction[0])
        heading_error = self._normalize_parallel_angle(
            line_angle - self.camera_forward_angle
        )
        selected_distance = self._cluster_min_distance(selected_cluster)
        line_length = self._line_length(point_on_line, direction, inliers)

        self.latest_line = {
            'stamp': time.monotonic(),
            'line_angle': line_angle,
            'heading_error': heading_error,
            'line_length': line_length,
        }

        self._publish_markers(
            msg.header,
            points,
            selected_cluster,
            point_on_line,
            direction,
            inliers
        )
        self._log_periodically(
            f"line_angle={math.degrees(line_angle):.1f} deg, "
            f"heading_error={math.degrees(heading_error):.1f} deg, "
            f"line_length={line_length:.2f} m, roi_points={len(points)}, "
            f"clusters={len(clusters)}, selected_points={len(selected_cluster)}, "
            f"selected_min_dist={selected_distance:.2f} m, inliers={len(inliers)}"
        )

    def odom_callback(self, msg):
        """Odometry에서 cmd_vel 이동/회전 확인용 yaw를 갱신합니다."""
        self.current_odom = msg
        self.current_yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)

    def amcl_pose_callback(self, msg):
        """map 좌표 target을 바라보기 위한 현재 map pose를 저장합니다."""
        pose = msg.pose.pose
        self.current_map_x = pose.position.x
        self.current_map_y = pose.position.y
        self.current_map_yaw = self._yaw_from_quaternion(pose.orientation)

    def _wait_for_ready(self):
        """라이다 직선, odom, 필요 시 amcl pose가 들어올 때까지 기다립니다."""
        start = time.monotonic()
        self.get_logger().info("라이다 직선, odom, 카메라 데이터를 기다립니다...")

        while rclpy.ok() and time.monotonic() - start < self.wait_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            line_is_fresh = (
                self.latest_line is not None
                and time.monotonic() - self.latest_line['stamp'] < 1.0
            )
            odom_ready = self.current_odom is not None and self.current_yaw is not None
            target_ready = (
                not self.use_target_pose
                or (
                    self.current_map_x is not None
                    and self.current_map_y is not None
                    and self.current_map_yaw is not None
                )
            )
            if line_is_fresh and odom_ready and target_ready:
                return True

        self.get_logger().error("필요한 센서 데이터를 시간 안에 받지 못했습니다.")
        return False

    def _wait_for_plate_detection(self, duration, label):
        """정지 상태에서 duration 동안 YOLO 번호판 성공 판정을 허용합니다."""
        self._stop_robot()
        self.accepted_plate_detected = False
        self.accept_plate_detection = True
        start = time.monotonic()
        self.get_logger().info(f"{label}: {duration:.1f}초 확인")

        while rclpy.ok() and time.monotonic() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            fresh_detection = (
                self.latest_plate_detected
                and self.latest_plate_stamp >= start
                and time.monotonic() - self.latest_plate_stamp < 1.0
            )
            if self.accepted_plate_detected or fresh_detection:
                self.accept_plate_detection = False
                self._stop_robot()
                self.get_logger().info(
                    f"번호판 탐지: confidence={self.latest_plate_confidence:.2f}"
                )
                return True

        self.accept_plate_detection = False
        self.get_logger().info("번호판 미탐지")
        return False

    def _face_target_pose(self):
        """AMCL map pose 기준으로 target_x/y 방향을 바라보도록 제자리 회전합니다."""
        if self.current_map_x is None or self.current_map_y is None:
            self.get_logger().error("AMCL pose가 없어 target 방향을 계산할 수 없습니다.")
            return False

        dx = self.target_x - self.current_map_x
        dy = self.target_y - self.current_map_y
        if math.hypot(dx, dy) < 1e-4:
            self.get_logger().warning("target 좌표가 현재 위치와 너무 가깝습니다.")
            return True

        target_yaw = math.atan2(dy, dx)
        rotate_angle = self._normalize_angle(target_yaw - self.current_map_yaw)
        self.get_logger().info(
            f"target_yaw={math.degrees(target_yaw):.1f} deg, "
            f"current_yaw={math.degrees(self.current_map_yaw):.1f} deg, "
            f"rotate={math.degrees(rotate_angle):.1f} deg"
        )
        return self._rotate_relative(rotate_angle)

    def _rotate_relative(self, target_angle):
        """odom yaw 변화를 보면서 target_angle만큼 제자리 회전합니다."""
        if abs(target_angle) <= self.angle_tolerance:
            self.get_logger().info("회전 오차가 작아서 회전을 생략합니다.")
            return True

        start_yaw = self.current_yaw
        direction = 1.0 if target_angle > 0.0 else -1.0
        timeout = max(4.0, abs(target_angle) / max(self.angular_speed, 1e-3) + 3.0)
        start = time.monotonic()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            rotated = self._normalize_angle(self.current_yaw - start_yaw)
            remaining = abs(target_angle) - abs(rotated)

            if remaining <= self.angle_tolerance:
                self._stop_robot()
                return True

            if time.monotonic() - start > timeout:
                self.get_logger().error("회전 timeout이 발생했습니다.")
                self._stop_robot()
                return False

            twist = Twist()
            twist.angular.z = direction * self.angular_speed
            self.cmd_pub.publish(twist)

        return False

    def _move_straight(self, distance):
        """odom 위치 변화량을 보면서 distance만큼 직선 이동합니다."""
        if abs(distance) <= self.distance_tolerance:
            return True

        start_pose = self.current_odom.pose.pose.position
        start_x = start_pose.x
        start_y = start_pose.y
        direction = 1.0 if distance > 0.0 else -1.0
        timeout = max(4.0, abs(distance) / max(self.linear_speed, 1e-3) + 3.0)
        start = time.monotonic()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if direction > 0.0 and self._front_obstacle_too_close():
                self.get_logger().error("전방 장애물이 가까워 이동을 중단합니다.")
                self._stop_robot()
                return False

            pose = self.current_odom.pose.pose.position
            moved = math.hypot(pose.x - start_x, pose.y - start_y)
            remaining = abs(distance) - moved

            if remaining <= self.distance_tolerance:
                self._stop_robot()
                return True

            if time.monotonic() - start > timeout:
                self.get_logger().error("직선 이동 timeout이 발생했습니다.")
                self._stop_robot()
                return False

            twist = Twist()
            twist.linear.x = direction * self.linear_speed
            self.cmd_pub.publish(twist)

        return False

    def _front_obstacle_too_close(self):
        """전진 중 안전거리 안에 라이다 점이 있으면 True를 반환합니다."""
        if not self.enable_front_safety_stop or self.latest_scan is None:
            return False

        msg = self.latest_scan
        for i, scan_range in enumerate(msg.ranges):
            if not math.isfinite(scan_range):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            if angle < self.front_safety_min or angle > self.front_safety_max:
                continue

            if self.roi_range_min <= scan_range <= self.front_safety_distance:
                return True

        return False

    def _detect_plate(self, img):
        """YOLO 추론 결과에서 번호판 class만 확인하고, 표시용 이미지를 반환합니다."""
        annotated = img.copy()
        results = self.model(img, verbose=False)
        plate_detected = False
        best_confidence = 0.0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                cls = int(box.cls[0])
                if cls != self.plate_class_id:
                    continue

                plate_detected = True
                best_confidence = max(best_confidence, confidence)
                label = self._get_class_label(cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"{label}: {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        if plate_detected:
            status_text = f"Plate detected: {best_confidence:.2f}"
            status_color = (0, 255, 0)
        else:
            status_text = 'Plate: not detected'
            status_color = (0, 0, 255)

        if not self.accept_plate_detection:
            status_text += ' / display only'

        cv2.putText(
            annotated,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        return plate_detected, best_confidence, annotated

    def _load_yolo_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        suffix = Path(model_path).suffix.lower()
        if suffix == '.pt':
            return YOLO(model_path)
        if suffix in ['.onnx', '.engine']:
            return YOLO(model_path, task='detect')
        raise ValueError(f"지원하지 않는 모델 형식입니다: {suffix}")

    def _get_class_label(self, cls):
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, f'class_{cls}')
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return f'class_{cls}'

    def _scan_to_roi_points(self, msg):
        points = []
        for i, scan_range in enumerate(msg.ranges):
            if not math.isfinite(scan_range):
                continue
            if scan_range < self.roi_range_min or scan_range > self.roi_range_max:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            if angle < self.roi_angle_min or angle > self.roi_angle_max:
                continue

            x = scan_range * math.cos(angle)
            y = scan_range * math.sin(angle)
            points.append((x, y))

        return np.array(points, dtype=np.float64)

    def _cluster_points(self, points):
        clusters = []
        current_cluster = [points[0]]

        for point in points[1:]:
            prev_point = current_cluster[-1]
            distance = np.linalg.norm(point - prev_point)

            if distance <= self.cluster_distance_threshold:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= self.min_points:
                    clusters.append(np.array(current_cluster, dtype=np.float64))
                current_cluster = [point]

        if len(current_cluster) >= self.min_points:
            clusters.append(np.array(current_cluster, dtype=np.float64))

        return clusters

    def _select_nearest_cluster(self, clusters):
        if not clusters:
            return None
        return min(clusters, key=self._cluster_min_distance)

    def _cluster_min_distance(self, cluster):
        distances = np.linalg.norm(cluster, axis=1)
        return float(np.min(distances))

    def _fit_line_ransac(self, points):
        best_inliers = None
        if len(points) < 2:
            return None

        for _ in range(self.ransac_iterations):
            idx1, idx2 = random.sample(range(len(points)), 2)
            p1 = points[idx1]
            p2 = points[idx2]
            direction = p2 - p1
            norm = np.linalg.norm(direction)

            if norm < 1e-6:
                continue

            direction = direction / norm
            distances = self._point_line_distances(points, p1, direction)
            inliers = points[distances < self.ransac_distance_threshold]

            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers

        if best_inliers is None or len(best_inliers) < self.min_points:
            return None

        return self._refine_line_with_pca(best_inliers)

    def _refine_line_with_pca(self, points):
        center = np.mean(points, axis=0)
        centered = points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        if direction[0] < 0:
            direction = -direction

        distances = self._point_line_distances(points, center, direction)
        inliers = points[distances < self.ransac_distance_threshold]

        if len(inliers) < self.min_points:
            inliers = points

        return center, direction, inliers

    def _point_line_distances(self, points, point_on_line, direction):
        relative = points - point_on_line
        return np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])

    def _line_length(self, point_on_line, direction, points):
        projections = (points - point_on_line) @ direction
        return float(np.max(projections) - np.min(projections))

    def _yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_parallel_angle(self, angle):
        while angle > math.pi / 2.0:
            angle -= math.pi
        while angle < -math.pi / 2.0:
            angle += math.pi
        return angle

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _move_direction_sign(self):
        return 1.0 if self.move_direction >= 0.0 else -1.0

    def _publish_markers(self, header, roi_points, selected_cluster, point_on_line, direction, inliers):
        roi_marker = self._make_points_marker(
            header=header,
            marker_id=0,
            points=roi_points,
            color=(0.45, 0.45, 0.45, 0.25),
            scale=0.015
        )
        selected_marker = self._make_points_marker(
            header=header,
            marker_id=1,
            points=selected_cluster,
            color=(0.0, 0.25, 1.0, 0.7),
            scale=0.025
        )
        inlier_marker = self._make_points_marker(
            header=header,
            marker_id=2,
            points=inliers,
            color=(1.0, 0.0, 0.0, 0.9),
            scale=0.03
        )

        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = 'plate_lidar_line'
        line_marker.id = 3
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.06
        line_marker.color.g = 1.0
        line_marker.color.a = 1.0

        projections = (inliers - point_on_line) @ direction
        p_start = point_on_line + direction * np.min(projections)
        p_end = point_on_line + direction * np.max(projections)
        line_marker.points = [
            self._to_point(p_start, z=0.10),
            self._to_point(p_end, z=0.10)
        ]

        self.marker_pub.publish(roi_marker)
        self.marker_pub.publish(selected_marker)
        self.marker_pub.publish(inlier_marker)
        self.marker_pub.publish(line_marker)

    def _make_points_marker(self, header, marker_id, points, color, scale):
        marker = Marker()
        marker.header = header
        marker.ns = 'plate_lidar_line'
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.points = [self._to_point(p) for p in points]
        return marker

    def _publish_delete_markers(self, header):
        for marker_id in [0, 1, 2, 3]:
            marker = Marker()
            marker.header = header
            marker.ns = 'plate_lidar_line'
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)

    def _to_point(self, xy, z=0.0):
        point = Point()
        point.x = float(xy[0])
        point.y = float(xy[1])
        point.z = z
        return point

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _sleep_with_spin(self, duration):
        end_time = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _log_periodically(self, text):
        now = time.monotonic()
        if now - self.last_log_time < self.log_period_sec:
            return
        self.last_log_time = now
        self.get_logger().info(text)

    def destroy_node(self):
        if self.show_yolo_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PlateLidarRunWithYoloTestNode()
    except Exception as exc:
        print(exc)
        if rclpy.ok():
            rclpy.shutdown()
        return

    try:
        node.run_sequence()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 입력으로 라이다+YOLO 테스트 노드를 종료합니다.")
    finally:
        node._stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
