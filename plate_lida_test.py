# plate_lida_test.py

import math
import random
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


class PlateLidarLineTestNode(Node):
    """LaserScan에서 차량 옆면처럼 보이는 직선을 추출해 각도를 확인하는 테스트 노드."""

    def __init__(self):
        super().__init__('plate_lidar_line_test_node')

        # LaserScan 입력 토픽입니다. robot4를 테스트 중이므로 기본값은 /robot4/scan입니다.
        self.declare_parameter('scan_topic', '/robot4/scan')

        # 라이다 점을 사용할 각도 범위입니다.
        # 0도는 로봇 정면, +90도는 왼쪽, -90도는 오른쪽입니다.
        # robot4 환경에서는 /scan 기준 -90도가 OAK-D 카메라 정면에 가까움
        # OAK-D RGB HFOV 약 69도 -> -90 ± 30도 = -120 ~ -60
        self.declare_parameter('roi_angle_min_deg', -120.0)
        self.declare_parameter('roi_angle_max_deg', -60.0)

        # 라이다 점을 사용할 거리 범위입니다.
        # 너무 가까운 잡음은 roi_range_min_m으로 제거하고,
        # 멀리 있는 벽/다른 차량은 roi_range_max_m으로 제거합니다.
        self.declare_parameter('roi_range_min_m', 0.15)
        self.declare_parameter('roi_range_max_m', 0.60)

        # cluster나 직선으로 인정하기 위한 최소 점 개수입니다.
        # 너무 낮으면 잡음도 직선으로 잡히고, 너무 높으면 짧은 차량 면을 놓칠 수 있습니다.
        self.declare_parameter('min_points', 12)

        # 라이다 점들을 cluster로 나눌 때, 인접 점 사이가 이 거리보다 멀면 다른 cluster로 봅니다.
        # 값이 크면 서로 다른 물체가 하나로 붙고, 값이 작으면 한 차량도 여러 조각으로 쪼개질 수 있습니다.
        self.declare_parameter('cluster_distance_threshold_m', 0.12)

        # RANSAC 반복 횟수입니다. 클수록 안정적일 수 있지만 계산량이 늘어납니다.
        self.declare_parameter('ransac_iterations', 80)

        # RANSAC에서 직선과 이 거리 이내인 점을 inlier로 인정합니다.
        # 값이 작으면 엄격하고, 값이 크면 휘거나 튄 점도 직선에 포함될 수 있습니다.
        self.declare_parameter('ransac_distance_threshold_m', 0.035)

        # 터미널 로그 출력 주기입니다. 너무 작으면 로그가 너무 많이 찍힙니다.
        self.declare_parameter('log_period_sec', 0.5)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.roi_angle_min = math.radians(
            self.get_parameter('roi_angle_min_deg').value
        )
        self.roi_angle_max = math.radians(
            self.get_parameter('roi_angle_max_deg').value
        )
        self.roi_range_min = self.get_parameter('roi_range_min_m').value
        self.roi_range_max = self.get_parameter('roi_range_max_m').value
        self.min_points = self.get_parameter('min_points').value
        self.cluster_distance_threshold = (
            self.get_parameter('cluster_distance_threshold_m').value
        )
        self.ransac_iterations = self.get_parameter('ransac_iterations').value
        self.ransac_distance_threshold = (
            self.get_parameter('ransac_distance_threshold_m').value
        )
        self.log_period_sec = self.get_parameter('log_period_sec').value

        self.last_log_time = 0.0

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )
        self.marker_pub = self.create_publisher(Marker, '/plate_lidar_line_marker', 10)

        self.get_logger().info(
            f"라이다 직선 추출 테스트 시작 "
            f"(scan: {self.scan_topic}, "
            f"angle: {math.degrees(self.roi_angle_min):.1f}~"
            f"{math.degrees(self.roi_angle_max):.1f} deg, "
            f"range: {self.roi_range_min:.2f}~{self.roi_range_max:.2f} m)"
        )

    def scan_callback(self, msg):
        points = self._scan_to_roi_points(msg)

        if len(points) < self.min_points:
            self._publish_delete_markers(msg.header)
            self._log_periodically(f"ROI points 부족: {len(points)}개")
            return

        clusters = self._cluster_points(points)
        selected_cluster = self._select_nearest_cluster(clusters)

        if selected_cluster is None:
            self._publish_delete_markers(msg.header)
            self._log_periodically(
                f"선택 가능한 cluster 없음: roi_points={len(points)}, "
                f"clusters={len(clusters)}"
            )
            return

        line = self._fit_line_ransac(selected_cluster)
        if line is None:
            self._publish_delete_markers(msg.header)
            self._log_periodically("직선을 안정적으로 찾지 못했습니다.")
            return

        point_on_line, direction, inliers = line
        line_angle = math.atan2(direction[1], direction[0])
        parallel_error = self._normalize_parallel_angle(line_angle)
        selected_distance = self._cluster_min_distance(selected_cluster)

        self._publish_markers(msg.header, points, selected_cluster, point_on_line, direction, inliers)
        self._log_periodically(
            f"line_angle={math.degrees(line_angle):.1f} deg, "
            f"parallel_error={math.degrees(parallel_error):.1f} deg, "
            f"roi_points={len(points)}, clusters={len(clusters)}, "
            f"selected_points={len(selected_cluster)}, "
            f"selected_min_dist={selected_distance:.2f} m, "
            f"inliers={len(inliers)}"
        )

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

        # 같은 직선은 180도 뒤집혀도 같으므로, 로그를 읽기 쉽게 x 양의 방향으로 맞춥니다.
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

    def _normalize_parallel_angle(self, angle):
        while angle > math.pi / 2.0:
            angle -= math.pi
        while angle < -math.pi / 2.0:
            angle += math.pi
        return angle

    def _publish_markers(self, header, roi_points, selected_cluster, point_on_line, direction, inliers):
        roi_marker = self._make_points_marker(
            header=header,
            marker_id=0,
            points=roi_points,
            color=(0.45, 0.45, 0.45, 0.45),
            scale=0.02
        )

        selected_marker = self._make_points_marker(
            header=header,
            marker_id=1,
            points=selected_cluster,
            color=(0.0, 0.25, 1.0, 1.0),
            scale=0.04
        )

        inlier_marker = self._make_points_marker(
            header=header,
            marker_id=2,
            points=inliers,
            color=(1.0, 0.0, 0.0, 1.0),
            scale=0.05
        )

        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = 'plate_lidar_line'
        line_marker.id = 3
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.08
        line_marker.color.g = 1.0
        line_marker.color.a = 1.0

        projections = (inliers - point_on_line) @ direction
        p_start = point_on_line + direction * np.min(projections)
        p_end = point_on_line + direction * np.max(projections)
        line_marker.points = [
            self._to_point(p_start, z=0.08),
            self._to_point(p_end, z=0.08)
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

    def _log_periodically(self, text):
        now = time.monotonic()
        if now - self.last_log_time < self.log_period_sec:
            return
        self.last_log_time = now
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = PlateLidarLineTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 입력으로 라이다 직선 테스트 노드를 종료합니다.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
