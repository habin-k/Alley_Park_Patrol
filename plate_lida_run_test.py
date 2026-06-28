# plate_lida_run_test.py
#
# 목적:
#   1. robot4 라이다(/robot4/scan)에서 차량 옆면처럼 보이는 직선을 찾습니다.
#   2. 그 직선과 로봇 카메라 정면 방향이 평행해지도록 cmd_vel로 회전합니다.
#   3. odom을 보면서 짧게 전진/후진하여 번호판이 보일 만한 위치를 탐색합니다.
#
# 주의:
#   이 코드는 Nav2 경로 계획이 아니라 cmd_vel 직접 제어입니다.
#   작은 거리 테스트용으로 쓰고, 주변 장애물과 비상정지 준비를 꼭 확인해야 합니다.

import math
import random
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


class PlateLidarRunTestNode(Node):
    """라이다로 차량면 직선을 찾고, 로봇을 그 직선과 평행하게 맞춘 뒤 짧게 이동하는 테스트 노드."""

    def __init__(self):
        super().__init__('plate_lidar_run_test_node')

        # robot4 테스트용 기본 토픽입니다.
        # scan_topic: 라이다 거리 데이터 입력
        # odom_topic: 회전/이동량을 확인할 로봇 위치 추정값
        # cmd_vel_topic: 로봇에 직접 속도 명령을 보내는 출력
        self.declare_parameter('scan_topic', '/robot4/scan')
        self.declare_parameter('odom_topic', '/robot4/odom')
        self.declare_parameter('cmd_vel_topic', '/robot4/cmd_vel')

        # robot4에서 OAK-D 카메라 정면은 /scan 각도 기준 대략 -90도였습니다.
        # 회전 오차는 "차량 직선 각도 - 카메라 정면 각도"로 계산합니다.
        self.declare_parameter('camera_forward_angle_deg', -90.0)

        # 카메라 정면 기준 ROI입니다. OAK-D RGB HFOV가 약 69도라서 -90 ± 35도 근처로 잡았습니다.
        self.declare_parameter('roi_angle_min_deg', -125.0)
        self.declare_parameter('roi_angle_max_deg', -55.0)
        self.declare_parameter('roi_range_min_m', 0.15)
        self.declare_parameter('roi_range_max_m', 0.70)

        # 라이다 점 분리와 직선 피팅 파라미터입니다.
        # min_points:
        #   cluster나 직선으로 인정할 최소 점 개수입니다.
        # cluster_distance_threshold_m:
        #   라이다 점 사이가 이 값보다 멀면 서로 다른 물체로 분리합니다.
        # ransac_distance_threshold_m:
        #   직선에서 이 거리 안에 들어온 점만 "직선 위 점"으로 인정합니다.
        # min_line_length_m:
        #   추출된 직선 길이가 너무 짧으면 차량면이 아니라고 보고 동작을 중단합니다.
        self.declare_parameter('min_points', 12)
        self.declare_parameter('cluster_distance_threshold_m', 0.12)
        self.declare_parameter('ransac_iterations', 80)
        self.declare_parameter('ransac_distance_threshold_m', 0.035)
        self.declare_parameter('min_line_length_m', 0.15)

        # 동작 파라미터입니다.
        # rotation_sign:
        #   실제 테스트에서 회전 방향이 반대로 나오면 -1.0으로 실행하면 됩니다.
        # angular_speed_rad_s / linear_speed_m_s:
        #   실제 로봇이 움직이는 속도입니다. 처음 테스트는 낮게 두는 편이 안전합니다.
        # angle_tolerance_deg / distance_tolerance_m:
        #   odom 기반 제어에서 "이 정도면 도착"이라고 판단하는 허용 오차입니다.
        # settle_time_sec:
        #   회전 직후 로봇 흔들림과 odom 갱신을 기다리는 짧은 안정화 시간입니다.
        # inspect_pause_sec:
        #   각 이동 후 카메라/RViz 화면을 사람이 확인할 수 있도록 멈춰 있는 시간입니다.
        self.declare_parameter('rotation_sign', 1.0)
        self.declare_parameter('angular_speed_rad_s', 0.25)
        self.declare_parameter('linear_speed_m_s', 0.05)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.declare_parameter('distance_tolerance_m', 0.015)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter('inspect_pause_sec', 10.0)

        # 라이다로 측정한 차량면 길이를 기준으로 이동하되, 이 거리 이상은 가지 않습니다.
        self.declare_parameter('max_search_distance_m', 1.50)

        # 전진 중 정면 가까운 장애물이 있으면 멈추는 간단한 안전장치입니다.
        self.declare_parameter('enable_front_safety_stop', True)
        self.declare_parameter('front_safety_min_deg', -125.0)
        self.declare_parameter('front_safety_max_deg', -55.0)
        self.declare_parameter('front_safety_distance_m', 0.25)

        self.declare_parameter('wait_timeout_sec', 10.0)
        self.declare_parameter('log_period_sec', 0.5)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
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
        self.inspect_pause = self.get_parameter('inspect_pause_sec').value
        self.max_search_distance = self.get_parameter('max_search_distance_m').value
        self.enable_front_safety_stop = self.get_parameter(
            'enable_front_safety_stop'
        ).value
        self.front_safety_min = math.radians(self.get_parameter('front_safety_min_deg').value)
        self.front_safety_max = math.radians(self.get_parameter('front_safety_max_deg').value)
        self.front_safety_distance = self.get_parameter('front_safety_distance_m').value
        self.wait_timeout = self.get_parameter('wait_timeout_sec').value
        self.log_period_sec = self.get_parameter('log_period_sec').value

        # 콜백에서 계속 갱신되는 최신 센서/상태값입니다.
        self.last_log_time = 0.0
        self.latest_scan = None        # 가장 최근 LaserScan. 안전거리 확인에도 사용합니다.
        self.latest_line = None        # 가장 최근 추출된 차량면 직선 정보입니다.
        self.current_odom = None       # 가장 최근 Odometry 메시지입니다.
        self.current_yaw = None        # odom quaternion에서 계산한 현재 yaw 각도입니다.

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
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(Marker, '/plate_lidar_line_marker', 10)

        self.get_logger().info(
            f"라이다 주행 테스트 시작 "
            f"(scan: {self.scan_topic}, odom: {self.odom_topic}, "
            f"cmd_vel: {self.cmd_vel_topic}, "
            f"roi: {math.degrees(self.roi_angle_min):.1f}~"
            f"{math.degrees(self.roi_angle_max):.1f} deg, "
            f"range: {self.roi_range_min:.2f}~{self.roi_range_max:.2f} m)"
        )

    def run_sequence(self):
        """전체 테스트 시나리오를 한 번 실행합니다.

        흐름:
            라이다/odom 준비 대기 -> 차량면 직선 검증 -> 평행 회전 -> 전후 offset 이동
        """
        if not self._wait_for_ready():
            self._stop_robot()
            return False

        # 회전 중에도 latest_line은 계속 갱신되므로, 시작 순간의 직선 정보를 복사해서 사용합니다.
        line = dict(self.latest_line)
        if line['line_length'] < self.min_line_length:
            self.get_logger().error(
                f"추출된 직선이 너무 짧습니다: {line['line_length']:.2f} m"
            )
            self._stop_robot()
            return False

        # heading_error는 "카메라 정면 방향"과 "차량 직선 방향"의 차이입니다.
        # rotation_sign은 실제 로봇 테스트에서 방향이 반대일 때 빠르게 보정하기 위한 계수입니다.
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

        target_distance = min(line['line_length'], self.max_search_distance)
        target_offset = -target_distance  # +면 전진, -면 후진입니다.

        self.get_logger().info(
            f"2단계: 평행 방향으로 전후 이동 테스트 "
            f"(line_length={line['line_length']:.2f} m, "
            f"target_offset={target_offset:+.2f} m)"
        )

        if abs(target_offset) > self.distance_tolerance:
            self.get_logger().info(f"{target_offset:+.2f} m 이동")
            if not self._move_straight(target_offset):
                self._stop_robot()
                return False

            self._stop_robot()
            self.get_logger().info(
                f"이 위치에서 teleop으로 차량 좌표 방향을 바라본 뒤 YOLO 확인하면 됩니다. "
                f"{self.inspect_pause:.1f}초 대기합니다."
            )
            self._sleep_with_spin(self.inspect_pause)

        self._stop_robot()
        self.get_logger().info("라이다 기반 평행 정렬/전후 이동 테스트 완료. 노드는 유지합니다.")
        return True

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
        # line_angle은 라이다 좌표계에서 본 차량면 직선의 방향입니다.
        # heading_error는 그 직선과 카메라 정면 방향이 얼마나 어긋났는지입니다.
        line_angle = math.atan2(direction[1], direction[0])
        heading_error = self._normalize_parallel_angle(
            line_angle - self.camera_forward_angle
        )
        selected_distance = self._cluster_min_distance(selected_cluster)
        line_length = self._line_length(point_on_line, direction, inliers)

        # run_sequence()가 사용할 수 있도록 가장 최근 직선 정보를 저장합니다.
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
        """Odometry에서 현재 위치와 yaw를 갱신합니다."""
        self.current_odom = msg
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _wait_for_ready(self):
        """라이다 직선과 odom이 모두 들어올 때까지 기다립니다."""
        start = time.monotonic()
        self.get_logger().info("라이다 직선과 odom 데이터를 기다립니다...")

        while rclpy.ok() and time.monotonic() - start < self.wait_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            line_is_fresh = (
                self.latest_line is not None
                and time.monotonic() - self.latest_line['stamp'] < 1.0
            )
            if line_is_fresh and self.current_odom is not None and self.current_yaw is not None:
                return True

        self.get_logger().error("라이다 직선 또는 odom 데이터를 시간 안에 받지 못했습니다.")
        return False

    def _rotate_relative(self, target_angle):
        """odom yaw 변화를 보면서 target_angle만큼 제자리 회전합니다."""
        if abs(target_angle) <= self.angle_tolerance:
            self.get_logger().info("회전 오차가 작아서 회전을 생략합니다.")
            return True

        # 시작 yaw를 기준으로, odom yaw 변화량이 target_angle에 도달할 때까지 회전합니다.
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

        # 시작 위치와 현재 위치의 유클리드 거리 차이로 이동량을 판단합니다.
        # distance가 음수이면 linear.x를 음수로 보내 후진합니다.
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

    def _scan_to_roi_points(self, msg):
        """LaserScan ranges를 ROI 내부의 2D 점(x, y) 배열로 변환합니다."""
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
        """ROI 점들을 거리 기준으로 여러 cluster로 분리합니다."""
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
        """차량 좌표 근처에 도착했다는 가정으로, 로봇에서 가장 가까운 cluster를 선택합니다."""
        if not clusters:
            return None

        return min(clusters, key=self._cluster_min_distance)

    def _cluster_min_distance(self, cluster):
        """cluster 안 점들 중 로봇 원점과 가장 가까운 거리입니다."""
        distances = np.linalg.norm(cluster, axis=1)
        return float(np.min(distances))

    def _fit_line_ransac(self, points):
        """RANSAC으로 튄 점을 제외하고 가장 그럴듯한 직선을 찾습니다."""
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
        """RANSAC inlier들을 PCA로 다시 정리해서 최종 직선 방향을 계산합니다."""
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
        """각 점과 직선 사이의 수직 거리를 계산합니다."""
        relative = points - point_on_line
        return np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])

    def _line_length(self, point_on_line, direction, points):
        """직선 방향으로 점들을 투영해서 라이다에 보이는 차량면 길이를 계산합니다."""
        projections = (points - point_on_line) @ direction
        return float(np.max(projections) - np.min(projections))

    def _normalize_parallel_angle(self, angle):
        """직선은 180도 뒤집혀도 같으므로 오차를 -90~90도 범위로 정규화합니다."""
        while angle > math.pi / 2.0:
            angle -= math.pi
        while angle < -math.pi / 2.0:
            angle += math.pi
        return angle

    def _normalize_angle(self, angle):
        """일반 yaw 차이를 -180~180도 범위로 정규화합니다."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _publish_markers(self, header, roi_points, selected_cluster, point_on_line, direction, inliers):
        """RViz에서 ROI 점, 선택 cluster, inlier, 최종 직선을 확인할 Marker를 발행합니다."""
        # 회색: ROI 전체 점
        roi_marker = self._make_points_marker(
            header=header,
            marker_id=0,
            points=roi_points,
            color=(0.45, 0.45, 0.45, 0.25),
            scale=0.015
        )

        # 파랑: 가장 가까운 cluster
        selected_marker = self._make_points_marker(
            header=header,
            marker_id=1,
            points=selected_cluster,
            color=(0.0, 0.25, 1.0, 0.7),
            scale=0.025
        )

        # 빨강: 선택 cluster 중 직선에 잘 맞는 RANSAC inlier
        inlier_marker = self._make_points_marker(
            header=header,
            marker_id=2,
            points=inliers,
            color=(1.0, 0.0, 0.0, 0.9),
            scale=0.03
        )

        # 초록: 최종 추정된 차량면 직선. 점들보다 위로 보이도록 z를 살짝 올립니다.
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
        """점 배열을 RViz POINTS Marker 메시지로 변환합니다."""
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
        """유효한 직선을 못 찾았을 때 이전 Marker를 RViz에서 지웁니다."""
        for marker_id in [0, 1, 2, 3]:
            marker = Marker()
            marker.header = header
            marker.ns = 'plate_lidar_line'
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)

    def _to_point(self, xy, z=0.0):
        """numpy 좌표를 visualization_msgs/Marker용 Point로 변환합니다."""
        point = Point()
        point.x = float(xy[0])
        point.y = float(xy[1])
        point.z = z
        return point

    def _stop_robot(self):
        """cmd_vel 0을 한 번 발행해서 로봇을 정지시킵니다."""
        self.cmd_pub.publish(Twist())

    def _sleep_with_spin(self, duration):
        """기다리는 동안에도 ROS 콜백을 처리해서 scan/odom이 계속 갱신되게 합니다."""
        end_time = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _log_periodically(self, text):
        """라이다 콜백 로그가 너무 많이 찍히지 않게 주기적으로만 출력합니다."""
        now = time.monotonic()
        if now - self.last_log_time < self.log_period_sec:
            return
        self.last_log_time = now
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = PlateLidarRunTestNode()

    try:
        node.run_sequence()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 입력으로 라이다 주행 테스트 노드를 종료합니다.")
    finally:
        node._stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
