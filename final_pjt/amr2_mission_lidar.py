#!/usr/bin/env python3

"""AMR2 mission node with lidar plate-position adjustment.

기존 amr2_mission.py와 plate_lida_run_test.py는 수정하지 않고,
Nav2 목표 도착 후 OCR 요청 전에 라이다 기반 평행 정렬/짧은 이동을 실행한다.
"""

import math
import random
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

try:
    from final_pjt.amr2_mission import Amr2Mission
except ImportError:
    from amr2_mission import Amr2Mission


class Amr2MissionLidar(Amr2Mission):
    """AMR2 mission flow plus lidar alignment before OCR."""

    ALIGNING_FOR_OCR = 'ALIGNING_FOR_OCR'

    def __init__(self):
        super().__init__()

        self.declare_parameter('scan_topic', '/robot4/scan')
        self.declare_parameter('odom_topic', '/robot4/odom')
        self.declare_parameter('cmd_vel_topic', '/robot4/cmd_vel')
        self.declare_parameter('camera_forward_angle_deg', -90.0)
        self.declare_parameter('roi_angle_min_deg', -125.0)
        self.declare_parameter('roi_angle_max_deg', -55.0)
        self.declare_parameter('roi_range_min_m', 0.15)
        self.declare_parameter('roi_range_max_m', 0.70)
        self.declare_parameter('min_points', 12)
        self.declare_parameter('cluster_distance_threshold_m', 0.12)
        self.declare_parameter('ransac_iterations', 80)
        self.declare_parameter('ransac_distance_threshold_m', 0.035)
        self.declare_parameter('min_line_length_m', 0.15)
        self.declare_parameter('rotation_sign', 1.0)
        self.declare_parameter('angular_speed_rad_s', 0.25)
        self.declare_parameter('linear_speed_m_s', 0.05)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.declare_parameter('distance_tolerance_m', 0.015)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter('inspect_pause_sec', 0.0)
        self.declare_parameter('max_search_distance_m', 1.50)
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
            self.get_parameter('camera_forward_angle_deg').value)
        self.roi_angle_min = math.radians(
            self.get_parameter('roi_angle_min_deg').value)
        self.roi_angle_max = math.radians(
            self.get_parameter('roi_angle_max_deg').value)
        self.roi_range_min = self.get_parameter('roi_range_min_m').value
        self.roi_range_max = self.get_parameter('roi_range_max_m').value
        self.min_points = self.get_parameter('min_points').value
        self.cluster_distance_threshold = self.get_parameter(
            'cluster_distance_threshold_m').value
        self.ransac_iterations = self.get_parameter('ransac_iterations').value
        self.ransac_distance_threshold = self.get_parameter(
            'ransac_distance_threshold_m').value
        self.min_line_length = self.get_parameter('min_line_length_m').value
        self.rotation_sign = self.get_parameter('rotation_sign').value
        self.angular_speed = self.get_parameter('angular_speed_rad_s').value
        self.linear_speed = self.get_parameter('linear_speed_m_s').value
        self.angle_tolerance = math.radians(
            self.get_parameter('angle_tolerance_deg').value)
        self.distance_tolerance = self.get_parameter('distance_tolerance_m').value
        self.settle_time = self.get_parameter('settle_time_sec').value
        self.inspect_pause = self.get_parameter('inspect_pause_sec').value
        self.max_search_distance = self.get_parameter('max_search_distance_m').value
        self.enable_front_safety_stop = self.get_parameter(
            'enable_front_safety_stop').value
        self.front_safety_min = math.radians(
            self.get_parameter('front_safety_min_deg').value)
        self.front_safety_max = math.radians(
            self.get_parameter('front_safety_max_deg').value)
        self.front_safety_distance = self.get_parameter(
            'front_safety_distance_m').value
        self.wait_timeout = self.get_parameter('wait_timeout_sec').value
        self.log_period_sec = self.get_parameter('log_period_sec').value

        self.last_lidar_log_time = 0.0
        self.latest_scan = None
        self.latest_line = None
        self.current_odom = None
        self.current_yaw = None
        self.align_thread = None

        self.create_subscription(LaserScan, self.scan_topic,
                                 self.scan_callback, 10)
        self.create_subscription(Odometry, self.odom_topic,
                                 self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(
            Marker, '/plate_lidar_line_marker', 10)

        self.get_logger().info(
            'AMR2 lidar-integrated mission node ready. '
            f'scan={self.scan_topic}, odom={self.odom_topic}, '
            f'cmd_vel={self.cmd_vel_topic}')

    def _nav_result(self, future):
        """Run lidar adjustment before OCR after reaching a target."""
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

        if self.state == self.GOING_TO_GOAL:
            self._start_lidar_alignment()
        elif self.state == self.GOING_TO_DOCK:
            self.get_logger().info('도킹 위치 도착')
            self._start_dock()

    def _start_lidar_alignment(self):
        """Start lidar alignment in a worker thread so callbacks keep updating."""
        self.state = self.ALIGNING_FOR_OCR
        self.align_thread = threading.Thread(
            target=self._run_lidar_then_ocr,
            daemon=True)
        self.align_thread.start()

    def _run_lidar_then_ocr(self):
        ok = False
        try:
            ok = self.run_lidar_sequence()
        except Exception as error:
            self.get_logger().error(f'라이다 정렬 중 예외 발생: {error}')
        finally:
            self._stop_robot()

        if ok:
            self.get_logger().info('라이다 정렬 완료. OCR 요청을 시작합니다.')
        else:
            self.get_logger().warning(
                '라이다 정렬 실패. 현재 위치에서 OCR 요청을 계속합니다.')

        self._start_ocr_wait()

    def run_lidar_sequence(self):
        """Lidar line detection, parallel rotation, and short forward move."""
        if not self._wait_for_lidar_ready():
            self._stop_robot()
            return False

        line = dict(self.latest_line)
        if line['line_length'] < self.min_line_length:
            self.get_logger().error(
                f"추출된 직선이 너무 짧습니다: {line['line_length']:.2f} m")
            self._stop_robot()
            return False

        rotate_angle = self.rotation_sign * line['heading_error']
        self.get_logger().info(
            f"라이다 1단계: 차량면과 평행 정렬 "
            f"(line_angle={math.degrees(line['line_angle']):.1f} deg, "
            f"heading_error={math.degrees(line['heading_error']):.1f} deg, "
            f"rotate_cmd={math.degrees(rotate_angle):.1f} deg)")
        if not self._rotate_relative(rotate_angle):
            self._stop_robot()
            return False

        self._sleep_without_spin(self.settle_time)

        target_distance = min(line['line_length'] * 2.0, self.max_search_distance)
        self.get_logger().info(
            f"라이다 2단계: 평행 방향 이동 "
            f"(line_length={line['line_length']:.2f} m, "
            f"target={target_distance:.2f} m)")
        if not self._move_straight(target_distance):
            self._stop_robot()
            return False

        self._stop_robot()
        self._sleep_without_spin(self.inspect_pause)
        return True

    def scan_callback(self, msg):
        """Compute a candidate vehicle-side line from LaserScan."""
        self.latest_scan = msg
        points = self._scan_to_roi_points(msg)

        if len(points) < self.min_points:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_lidar_periodically(f"ROI points 부족: {len(points)}개")
            return

        clusters = self._cluster_points(points)
        selected_cluster = self._select_nearest_cluster(clusters)

        if selected_cluster is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_lidar_periodically(
                f"선택 가능한 cluster 없음: roi_points={len(points)}, "
                f"clusters={len(clusters)}")
            return

        line = self._fit_line_ransac(selected_cluster)
        if line is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_lidar_periodically("직선을 안정적으로 찾지 못했습니다.")
            return

        point_on_line, direction, inliers = line
        line_angle = math.atan2(direction[1], direction[0])
        heading_error = self._normalize_parallel_angle(
            line_angle - self.camera_forward_angle)
        selected_distance = self._cluster_min_distance(selected_cluster)
        line_length = self._line_length(point_on_line, direction, inliers)

        self.latest_line = {
            'stamp': time.monotonic(),
            'line_angle': line_angle,
            'heading_error': heading_error,
            'line_length': line_length,
            'selected_distance': selected_distance,
        }

        self._publish_markers(
            msg.header, points, selected_cluster, point_on_line,
            direction, inliers)
        self._log_lidar_periodically(
            f"line_angle={math.degrees(line_angle):.1f} deg, "
            f"heading_error={math.degrees(heading_error):.1f} deg, "
            f"line_length={line_length:.2f} m, roi_points={len(points)}, "
            f"clusters={len(clusters)}, selected_points={len(selected_cluster)}, "
            f"selected_min_dist={selected_distance:.2f} m, inliers={len(inliers)}")

    def odom_callback(self, msg):
        """Store odometry and current yaw."""
        self.current_odom = msg
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _wait_for_lidar_ready(self):
        start = time.monotonic()
        self.get_logger().info("라이다 직선과 odom 데이터를 기다립니다...")

        while rclpy.ok() and time.monotonic() - start < self.wait_timeout:
            line_is_fresh = (
                self.latest_line is not None
                and time.monotonic() - self.latest_line['stamp'] < 1.0
            )
            if (line_is_fresh and self.current_odom is not None
                    and self.current_yaw is not None):
                return True
            time.sleep(0.05)

        self.get_logger().error(
            "라이다 직선 또는 odom 데이터를 시간 안에 받지 못했습니다.")
        return False

    def _rotate_relative(self, target_angle):
        if abs(target_angle) <= self.angle_tolerance:
            self.get_logger().info("회전 오차가 작아서 회전을 생략합니다.")
            return True

        if self.current_yaw is None:
            return False

        start_yaw = self.current_yaw
        direction = 1.0 if target_angle > 0.0 else -1.0
        timeout = max(4.0, abs(target_angle) / max(self.angular_speed, 1e-3) + 3.0)
        start = time.monotonic()

        while rclpy.ok():
            if self.current_yaw is None:
                time.sleep(0.05)
                continue

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
            time.sleep(0.05)

        return False

    def _move_straight(self, distance):
        if abs(distance) <= self.distance_tolerance:
            return True

        if self.current_odom is None:
            return False

        start_pose = self.current_odom.pose.pose.position
        start_x = start_pose.x
        start_y = start_pose.y
        direction = 1.0 if distance > 0.0 else -1.0
        timeout = max(4.0, abs(distance) / max(self.linear_speed, 1e-3) + 3.0)
        start = time.monotonic()

        while rclpy.ok():
            if direction > 0.0 and self._front_obstacle_too_close():
                self.get_logger().error("전방 장애물이 가까워 이동을 중단합니다.")
                self._stop_robot()
                return False

            if self.current_odom is None:
                time.sleep(0.05)
                continue

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
            time.sleep(0.05)

        return False

    def _front_obstacle_too_close(self):
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
        return np.abs(relative[:, 0] * direction[1]
                      - relative[:, 1] * direction[0])

    def _line_length(self, point_on_line, direction, points):
        projections = (points - point_on_line) @ direction
        return float(np.max(projections) - np.min(projections))

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

    def _publish_markers(self, header, roi_points, selected_cluster,
                         point_on_line, direction, inliers):
        roi_marker = self._make_points_marker(
            header, 0, roi_points, (0.45, 0.45, 0.45, 0.25), 0.015)
        selected_marker = self._make_points_marker(
            header, 1, selected_cluster, (0.0, 0.25, 1.0, 0.7), 0.025)
        inlier_marker = self._make_points_marker(
            header, 2, inliers, (1.0, 0.0, 0.0, 0.9), 0.03)

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
            self._to_point(p_end, z=0.10),
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

    def _sleep_without_spin(self, duration):
        end_time = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end_time:
            time.sleep(0.05)

    def _log_lidar_periodically(self, text):
        now = time.monotonic()
        if now - self.last_lidar_log_time < self.log_period_sec:
            return
        self.last_lidar_log_time = now
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = Amr2MissionLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
