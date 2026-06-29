# amr_move_peter.py

import rclpy
from rclpy.node import Node
import json
import math 
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav2_simple_commander.robot_navigator import TaskResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Path
from transforms3d.euler import euler2quat
from ultralytics import YOLO


from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from geometry_msgs.msg import PoseWithCovarianceStamped

from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy

amcl_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# INITIAL_X = 0.0
# INITIAL_Y = 0.11
# INITIAL_YAW = 0.0

# PATROL TARGET
PATROL_TARGETS = [
            (-0.719, 0.324, 0.0),          
            (-3.4, 0.257, 0.0),
            (-5.2, 0.118, 0.0),
            (-4.8, 3.94, 0.0),
            (-0.056, 3.47, 0.0)

        ]

# amr_move.py (집 테스트용 임시 좌표 - 로봇 기준 앞뒤양옆 1~2m)
# PATROL_TARGETS = [
#     (1.0, 0.0, 0.0),          
#     (1.0, 1.0, 0.0),
#     (0.0, 1.0, 0.0),
#     (0.0, 0.0, 0.0)
# ]

class Amrmove(Node):
    #-----------
    # 초기화
    #-----------
    def __init__(self):
        super().__init__('amr_move', namespace='/robot2')
        self.navigator = TurtleBot4Navigator(namespace='/robot2')
        
        self.default_paths = []
        self.mission_paths = []
        self.return_paths = []
        self.mission_targets = []
        self.finished_target = None
        self.path_index = 0
        self.mode = "PATROL"
        self.current_task = False
        self.current_pose = None
        self.path_ranges = []
        self.patrol_waypoints = []

        self.targets = []
        self.pending_targets =[]
        self.known_locations = []
        self.DUPLICATE_THRESHOLD = 0.05

        # -------------------------
        # 번호판 재탐색/전송 설정
        # -------------------------
        # 미션 좌표에 도착했을 때 바로 YOLO로 번호판을 확인하고,
        # 안 보이면 라이다 직선으로 차량 옆면과 평행하게 정렬한 뒤 전방 이동합니다.
        self.declare_parameter('plate_recovery_enabled', True)
        self.declare_parameter(
            'model_path',
            '/home/rokey/rokey_ws/src/final_pjt/final_pjt_peter/semi_allimages_v5n.pt'
        )
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('plate_class_id', 2)
        self.declare_parameter('show_yolo_window', True)
        self.declare_parameter('yolo_min_period_sec', 0.20)
        self.declare_parameter('initial_yolo_check_sec', 1.5)
        self.declare_parameter('final_yolo_check_sec', 3.0)
        self.declare_parameter('retry_yolo_check_sec', 3.0)

        # amr_move_peter는 namespace=/robot2 노드라서 상대 토픽명은 /robot2/... 로 해석됩니다.
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('camera_topic', 'oakd/rgb/image_raw/compressed')
        self.declare_parameter('plate_image_topic', 'target_plate_image')
        self.declare_parameter('plate_id_topic', 'plate_id')

        # robot4 실험에서 OAK-D 정면은 /scan 기준 대략 -90도였고,
        # robot2에서도 같은 장착 방향이면 이 값을 그대로 씁니다.
        self.declare_parameter('camera_forward_angle_deg', -90.0)
        self.declare_parameter('roi_angle_min_deg', -120.0)
        self.declare_parameter('roi_angle_max_deg', -60.0)
        self.declare_parameter('roi_range_min_m', 0.15)
        self.declare_parameter('roi_range_max_m', 0.70)
        self.declare_parameter('min_points', 12)
        self.declare_parameter('cluster_distance_threshold_m', 0.12)
        self.declare_parameter('ransac_iterations', 80)
        self.declare_parameter('ransac_distance_threshold_m', 0.035)
        self.declare_parameter('min_line_length_m', 0.15)

        # 1차는 초록 선 길이의 1.5배, 실패 시 추가 2.0배를 더 가서 총 3.5배 위치를 봅니다.
        self.declare_parameter('rotation_sign', 1.0)
        self.declare_parameter('angular_speed_rad_s', 0.25)
        self.declare_parameter('linear_speed_m_s', 0.05)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.declare_parameter('distance_tolerance_m', 0.015)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter('first_move_multiplier', 3.0)
        self.declare_parameter('retry_total_move_multiplier', 6.0)
        self.declare_parameter('max_search_distance_m', 1.50)
        self.declare_parameter('enable_front_safety_stop', True)
        self.declare_parameter('front_safety_min_deg', -120.0)
        self.declare_parameter('front_safety_max_deg', -60.0)
        self.declare_parameter('front_safety_distance_m', 0.25)
        self.declare_parameter('plate_wait_timeout_sec', 10.0)
        self.declare_parameter('plate_log_period_sec', 0.5)

        self._load_plate_parameters()
        
        self.sub_wb_xyz = self.create_subscription(String, '/webcam_objects/map_detections', self.xyz_callback, 10)
        self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, "/robot2/amcl_pose", self.amcl_callback, amcl_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, self.camera_topic, self.image_callback, 10)
        self.pub_xyzr_amr2 = self.create_publisher(String, '/a_to_b', 10)
        self.pub_target_plate_image = self.create_publisher(
            CompressedImage,
            self.plate_image_topic,
            10
        )
        self.pub_plate_id = self.create_publisher(String, self.plate_id_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_targets = self.create_publisher(MarkerArray, "/robot2/mission_targets", 10)
        self.pub_debug_path = self.create_publisher(Path, "/robot2/debug_path", 10)
        self.marker_pub = self.create_publisher(Marker, 'plate_lidar_line_marker', 10)

        self.get_logger().info("amcl subscription created")
        self.get_logger().info(f"Resolved topic = {self.resolve_topic_name('/robot2/amcl_pose')}")
        self.get_logger().info(
            f"plate recovery topics: scan={self.scan_topic}, odom={self.odom_topic}, "
            f"camera={self.camera_topic}, cmd_vel={self.cmd_vel_topic}"
        )
        
        self.wait_until = None

        # 라이다/odom/YOLO 최신 상태입니다. 미션 도착 후 번호판 확인 루틴에서 사용합니다.
        self.latest_scan = None
        self.latest_line = None
        self.current_odom = None
        self.current_yaw = None
        self.last_log_time = 0.0
        self.last_move_stopped_by_obstacle = False

        self.bridge = CvBridge()
        self.model = None
        self.class_names = {}
        self.window_name = 'amr_move_plate_yolo_view'
        self.last_yolo_time = 0.0
        self.latest_plate_detected = False
        self.latest_plate_stamp = 0.0
        self.latest_plate_confidence = 0.0
        self.accept_plate_detection = False
        self.accepted_plate_detected = False
        self.accepted_plate_image_msg = None

        if self.plate_recovery_enabled:
            self._init_yolo_model()

    def _load_plate_parameters(self):
        """번호판 재탐색에 필요한 ROS 파라미터를 멤버 변수로 읽어옵니다."""
        self.plate_recovery_enabled = self.get_parameter('plate_recovery_enabled').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.plate_class_id = self.get_parameter('plate_class_id').value
        self.show_yolo_window = self.get_parameter('show_yolo_window').value
        self.yolo_min_period = self.get_parameter('yolo_min_period_sec').value
        self.initial_yolo_check_sec = self.get_parameter('initial_yolo_check_sec').value
        self.final_yolo_check_sec = self.get_parameter('final_yolo_check_sec').value
        self.retry_yolo_check_sec = self.get_parameter('retry_yolo_check_sec').value

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.plate_image_topic = self.get_parameter('plate_image_topic').value
        self.plate_id_topic = self.get_parameter('plate_id_topic').value

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
        self.first_move_multiplier = self.get_parameter('first_move_multiplier').value
        self.retry_total_move_multiplier = self.get_parameter(
            'retry_total_move_multiplier'
        ).value
        self.max_search_distance = self.get_parameter('max_search_distance_m').value
        self.enable_front_safety_stop = self.get_parameter(
            'enable_front_safety_stop'
        ).value
        self.front_safety_min = math.radians(self.get_parameter('front_safety_min_deg').value)
        self.front_safety_max = math.radians(self.get_parameter('front_safety_max_deg').value)
        self.front_safety_distance = self.get_parameter('front_safety_distance_m').value
        self.plate_wait_timeout = self.get_parameter('plate_wait_timeout_sec').value
        self.plate_log_period_sec = self.get_parameter('plate_log_period_sec').value

    def _init_yolo_model(self):
        """YOLO 모델을 로드합니다. 실패하면 번호판 재탐색만 비활성화합니다."""
        model_path = self.get_parameter('model_path').value

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            suffix = Path(model_path).suffix.lower()
            if suffix == '.pt':
                self.model = YOLO(model_path)
            elif suffix in ['.onnx', '.engine']:
                self.model = YOLO(model_path, task='detect')
            else:
                raise ValueError(f"지원하지 않는 모델 형식입니다: {suffix}")

            self.class_names = self.model.names
            self.get_logger().info(f"YOLO class names: {self.class_names}")
        except Exception as exc:
            self.model = None
            self.plate_recovery_enabled = False
            self.get_logger().error(f"YOLO 모델 로드 실패. 번호판 재탐색 비활성화: {exc}")


    #-----------
    # 콜백
    #-----------
    def amcl_callback(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

    def odom_callback(self, msg):
        """cmd_vel 이동/회전량 확인용 odom yaw를 저장합니다."""
        self.current_odom = msg
        self.current_yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        """라이다 ROI에서 가장 가까운 cluster를 뽑고 차량 옆면 직선을 추정합니다."""
        self.latest_scan = msg
        points = self._scan_to_roi_points(msg)

        if len(points) < self.min_points:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_plate_periodically(f"ROI points 부족: {len(points)}개")
            return

        clusters = self._cluster_points(points)
        selected_cluster = self._select_nearest_cluster(clusters)
        if selected_cluster is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_plate_periodically(
                f"선택 가능한 cluster 없음: roi_points={len(points)}, clusters={len(clusters)}"
            )
            return

        line = self._fit_line_ransac(selected_cluster)
        if line is None:
            self.latest_line = None
            self._publish_delete_markers(msg.header)
            self._log_plate_periodically("직선을 안정적으로 찾지 못했습니다.")
            return

        point_on_line, direction, inliers = line
        line_angle = math.atan2(direction[1], direction[0])
        heading_error = self._normalize_parallel_angle(
            line_angle - self.camera_forward_angle
        )
        line_length = self._line_length(point_on_line, direction, inliers)

        self.latest_line = {
            'stamp': time.monotonic(),
            'line_angle': line_angle,
            'heading_error': heading_error,
            'line_length': line_length,
        }

        self._publish_plate_markers(
            msg.header,
            points,
            selected_cluster,
            point_on_line,
            direction,
            inliers
        )
        self._log_plate_periodically(
            f"line_angle={math.degrees(line_angle):.1f} deg, "
            f"heading_error={math.degrees(heading_error):.1f} deg, "
            f"line_length={line_length:.2f} m, roi_points={len(points)}, "
            f"clusters={len(clusters)}, selected_points={len(selected_cluster)}, "
            f"inliers={len(inliers)}"
        )

    def image_callback(self, msg):
        """카메라 compressed image를 YOLO에 넣고, 번호판이 보이면 전송용 이미지를 저장합니다."""
        if self.model is None:
            return

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

            # 성공 판정 구간에서만 target_plate_image로 보낼 이미지를 저장합니다.
            if self.accept_plate_detection:
                plate_image_msg = self._cv2_to_compressed_msg(
                    annotated,
                    msg.header
                )
                if plate_image_msg is not None:
                    self.accepted_plate_detected = True
                    self.accepted_plate_image_msg = plate_image_msg
                    self._stop_robot()

        if self.show_yolo_window:
            cv2.imshow(self.window_name, annotated)
            cv2.waitKey(1)

    def xyz_callback(self, msg):
        new_targets = self.json_to_dic(msg)
        if not new_targets or len(new_targets) == 0:
            return
        
        added_new = False

        for target in new_targets:
            tx = target["x"]
            ty = target["y"]

            if not self.is_duplicate(tx, ty):
                self.known_locations.append((tx, ty))
                self.pending_targets.append(target)
                added_new = True

        if not added_new:
            return
        if self.mode == "MISSION":
            self.get_logger().info("새좌표 들어옴")
            return
        self.start_pending_mission()

    def start_pending_mission(self):
        if not self.pending_targets:
            return
        
        self.targets = self.pending_targets.copy()
        self.pending_targets.clear()

        self.cancel_task()
        self.sort_xyzr()
        self.path_remake()

    # 디버깅용
    def publish_targets(self):
        marker_array = MarkerArray()

        for i, target in enumerate(self.targets):
            marker = Marker()

            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "mission_targets"
            marker.id = i

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = target["x"]
            marker.pose.position.y = target["y"]
            marker.pose.position.z = 0.1

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.pub_targets.publish(marker_array)

    #-----------
    # 데이터 처리
    #-----------
    def is_duplicate(self, target_x, target_y):
        for kx, ky in self.known_locations:
            dist = math.hypot(kx - target_x, ky - target_y)
            if dist <= self.DUPLICATE_THRESHOLD:
                return True
        return False
    
    def json_to_dic(self, msg):
        try:
            data = json.loads(msg.data)

            system_time = data["system_time"]

            targets = []

            OFFSET_X_1 = -0.14
            OFFSET_Y_1 = -0.465

            OFFSET_X_2 = 0.32
            OFFSET_Y_2 = 0.931

            for obj in data["objects"]:
                event_id = obj["event_id"]
                zone = int(obj["zone"])
                raw_x = obj["x"]
                raw_y = obj["y"]
                if zone == 1 :
                    final_x = raw_x + OFFSET_X_1
                    final_y = raw_y + OFFSET_Y_1
                
                elif zone == 2:
                    final_x = raw_x + OFFSET_X_2
                    final_y = raw_y + OFFSET_Y_2
                
                elif zone ==3:
                    self.get_logger().warn(f"another zone not save {zone}")
                    continue

                elif zone == 4:
                    self.get_logger().warn(f"another zone not save {zone}")
                    continue

                else :
                    self.get_logger().warn(f"이상한 존 들어옴 {zone}")
                    continue
                
                # else:
                #     self.get_logger().warn(f"another zone not save {zone}")
                #     continue
                
                targets.append({
                    "event_id": event_id,
                    "zone": zone,
                    # raw_x/raw_y는 실제 차량 좌표입니다.
                    # x/y는 offset을 적용한 로봇 접근 좌표라서, 차량을 바라볼 때는 raw 좌표를 씁니다.
                    "car_x": raw_x,
                    "car_y": raw_y,
                    "x": final_x,
                    "y": final_y,
                    "time": system_time
                })
                
            return targets
            
        except Exception as e:
            self.get_logger().error(f"json: {e}")
            return []
    
    def publish_target_to_amr2(self, target): 

        data = {
            "event_id": target["event_id"],
            "zone": target["zone"],

            "x": self.current_pose.pose.position.x,
            "y": self.current_pose.pose.position.y,
            "orientation": {
                "x": self.current_pose.pose.orientation.x,
                "y": self.current_pose.pose.orientation.y,
                "z": self.current_pose.pose.orientation.z,
                "w": self.current_pose.pose.orientation.w
            }
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub_xyzr_amr2.publish(msg)
        self.get_logger().info(f"Publish: {data}")

    def publish_plate_result(self, target):
        """번호판 탐지 성공 시 이미지와 event_id/zone을 각각 발행합니다."""
        if self.accepted_plate_image_msg is None:
            self.get_logger().warning("번호판 이미지가 없어 plate result를 발행하지 않습니다.")
            return False

        target_msg = String()
        target_msg.data = json.dumps({
            'event_id': str(target["event_id"]),
            'zone': str(target["zone"]),
        })

        self.pub_target_plate_image.publish(self.accepted_plate_image_msg)
        self.pub_plate_id.publish(target_msg)
        self.get_logger().info(
            f"번호판 이미지/ID 발행 완료: {target_msg.data}"
        )
        return True
     
    #-----------
    # Utility
    #-----------   
    def make_pose(self, frame_id, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        qw, qx, qy, qz = euler2quat(0.0, 0.0, yaw)

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def target_heading_yaw(self, target):
        """접근 좌표 x/y에서 실제 차량 좌표 car_x/car_y를 바라보는 yaw를 계산합니다."""
        car_x = target.get("car_x", target["x"])
        car_y = target.get("car_y", target["y"])
        dx = car_x - target["x"]
        dy = car_y - target["y"]

        if math.hypot(dx, dy) < 1e-4:
            return 0.0

        return math.atan2(dy, dx)
    
    def nearest_wp(self, pose):
        if not hasattr(self, "patrol_waypoints"):
            return 0

        min_distance = float("inf")
        nearest_index = 0

        for i, wp in enumerate(self.patrol_waypoints):
            dx = pose.pose.position.x - wp.pose.position.x
            dy = pose.pose.position.y - wp.pose.position.y
            distance = dx*dx + dy*dy

            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index
    
    def nearest_patrol_path(self):
        if self.current_pose is None:
            return 0
        wp = self.nearest_wp(self.current_pose)

        for i, (start, end) in enumerate(self.path_ranges):
            if start <= wp <= end:
                return i
        
        return 0
    
    #-----------
    # 경로 생성
    #-----------    
    def default_path(self, start):
        if self.current_pose is None:
            self.get_logger().warn("amcl not received")
            return 

        self.default_paths = []
        self.patrol_waypoints = []
        self.path_length = []
        
        targets = PATROL_TARGETS

        self.path_ranges.clear()
        wp_start = 0

        for i in range(len(targets)):
            goal = self.make_pose(
                "map",
                targets[(i+1) % len(targets)][0],
                targets[(i+1) % len(targets)][1],
                targets[(i+1) % len(targets)][2]
            )
            path = self.navigator.getPath(start, goal)

            if path is not None:
                self.pub_debug_path.publish(path)

            if path is None:
                self.get_logger().error(f"path{i} no !! ")
                

            self.default_paths.append(path)
            self.patrol_waypoints.extend(path.poses)
            wp_end = wp_start + len(path.poses) -1
            self.path_ranges.append((wp_start, wp_end))
            wp_start = wp_end +1
            self.get_logger().info(f"path{i+1}: {len(path.poses)} waypoints")
        
            start = goal
        
        self.path_length.append(0.0)

        for i in range(1, len(self.patrol_waypoints)):
            p1 = self.patrol_waypoints[i - 1].pose.position
            p2 = self.patrol_waypoints[i].pose.position

            d = math.hypot(
                p2.x - p1.x,
                p2.y - p1.y
            )
            self.path_length.append(self.path_length[-1] + d)

        self.get_logger().info(f"Total waypoints: {len(self.patrol_waypoints)}")
        self.get_logger().info(f"total patrol lengrh: {self.path_length[-1]:.2f}")
        self.path_index = 0

    def sort_xyzr(self):
        if self.current_pose is None or not hasattr(self, "path_length") or len(self.path_length) == 0:
            self.get_logger().warn("Patrol path not generated yet. Skiping sort.")
            return
        if self.current_pose is None:
            return 
        
        robot_pose = self.current_pose
        robot_wp = self.nearest_wp(robot_pose)
        robot_s = self.path_length[robot_wp]
        total_length = self.path_length[-1]

        for target in self.targets:
            target_pose = self.make_pose("map", target["x"], target["y"], 0.0)
            target_wp = self.nearest_wp(target_pose)
            target["wp"] = target_wp
            target_s = self.path_length[target_wp]
            distance = target_s - robot_s
            
            if distance < 0:
                distance += total_length
            
            target["distance"] = distance
        
        self.targets.sort(key=lambda t: t["distance"])

    def path_remake(self):
        if self.current_pose is None:
            self.get_logger().warn("amcl not received")
            return
        
        self.mission_paths = []
        self.mission_targets = []

        start = self.current_pose

        for target in self.targets:
            # 미션 도착 시점부터 차량 쪽을 보게 하려고 goal yaw도 차량 좌표 방향으로 넣습니다.
            goal_yaw = self.target_heading_yaw(target)
            goal = self.make_pose("map", target["x"], target["y"], goal_yaw)
            path = self.navigator.getPath(start, goal)

            if path is None:
                self.get_logger().warn(f"{target['event_id']} path make fail")
                continue
            
            self.pub_debug_path.publish(path)
            self.mission_paths.append(path)
            self.mission_targets.append(target)
            self.publish_targets()

            start = goal
        
        if len(self.mission_paths) == 0:
            self.get_logger().warn("all mission fail go to patrol")
            self.return_to_patrol()
            return

        self.path_index = 0
        self.mode = "MISSION"

    def return_to_patrol(self):
        self.return_paths = []
        self.next_patrol_path = self.nearest_patrol_path()

        nearest_wp = self.nearest_wp(self.current_pose)

        start = self.current_pose
        wp_pose = self.patrol_waypoints[nearest_wp]

        connect = self.navigator.getPath(start, wp_pose)
        
        if connect is not None:
            self.return_paths.append(connect)
        
        start_wp, end_wp = self.path_ranges[self.next_patrol_path]
        
        remain_poses = self.patrol_waypoints[nearest_wp+1:end_wp+1]
        if len(remain_poses) > 0:
            remain = Path()
            remain.header = self.default_paths[self.next_patrol_path].header
            remain.poses = remain_poses
            self.return_paths.append(remain)
        if len(self.return_paths) == 0:
            self.get_logger().warn("reutn fail go to next patrol")
            self.mode = "PATROL"
            self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)
            self.known_locations.clear()
            return

        self.mode = "RETURN"
        self.path_index = 0
        self.known_locations.clear()

    #-----------
    # Plate recovery
    #-----------
    def run_plate_recovery_for_target(self, target):
        """미션 도착 후 번호판을 확인하고, 실패하면 라이다 기반 보정 주행을 수행합니다."""
        if not self.plate_recovery_enabled or self.model is None:
            self.get_logger().warning("번호판 재탐색이 비활성화되어 있습니다.")
            return False

        # 차량을 바라볼 때는 offset이 적용된 접근 좌표가 아니라 원래 차량 좌표를 사용합니다.
        self.target_x = target.get("car_x", target["x"])
        self.target_y = target.get("car_y", target["y"])
        self._reset_plate_detection_state()
        self._stop_robot()
        self._sleep_with_spin(0.2)

        # 1. 도착 직후 이미 번호판이 보이면 이동 보정 없이 바로 성공입니다.
        if self.initial_yolo_check_sec > 0.0:
            if self._wait_for_plate_detection(
                self.initial_yolo_check_sec,
                "도착 직후 번호판 확인"
            ):
                self.get_logger().info("도착 자세에서 번호판 탐지 성공.")
                return True

        # 2. 번호판이 안 보이면 라이다 직선/odom/map pose가 준비될 때까지 기다립니다.
        if not self._wait_for_plate_recovery_ready():
            self._stop_robot()
            return False

        line = dict(self.latest_line)
        if line['line_length'] < self.min_line_length:
            self.get_logger().warning(
                f"추출된 차량면 직선이 너무 짧습니다: {line['line_length']:.2f} m"
            )
            self._stop_robot()
            return False

        # 3. 차량 옆면 직선과 카메라 정면 방향이 평행해지도록 회전합니다.
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
        parallel_odom_yaw = self.current_yaw

        # 4. 1차 탐색: 초록 선 길이의 1.5배만큼 전진한 뒤 차량 좌표를 바라봅니다.
        first_distance = min(
            line['line_length'] * self.first_move_multiplier,
            self.max_search_distance
        )
        self.get_logger().info(
            f"2단계: 1차 전진 "
            f"(line_length={line['line_length']:.2f} m, "
            f"multiplier={self.first_move_multiplier:.2f}, "
            f"distance={first_distance:.2f} m)"
        )
        if first_distance > self.distance_tolerance:
            if not self._move_straight(first_distance):
                self._stop_robot()
                return False

        self._stop_robot()
        self._sleep_with_spin(self.settle_time)

        if not self._face_target_pose():
            self._stop_robot()
            return False

        self._sleep_with_spin(self.settle_time)
        if self.final_yolo_check_sec > 0.0:
            if self._wait_for_plate_detection(
                self.final_yolo_check_sec,
                "1차 이동 후 번호판 확인"
            ):
                self.get_logger().info("1차 이동 위치에서 번호판 탐지 성공.")
                return True

        # 전진 중 장애물을 만나 멈춘 경우, 같은 방향으로 더 가는 2차 탐색은 위험하므로 포기합니다.
        if self.last_move_stopped_by_obstacle:
            self.get_logger().warning(
                "1차 전진 중 장애물로 멈췄고 번호판도 보이지 않아 이번 target을 포기합니다."
            )
            self._stop_robot()
            return False

        # 5. 2차 탐색: target을 바라본 자세에서 다시 평행 자세로 돌아온 뒤,
        # 현재 위치에서 추가로 2.0배 전진해 총 3.5배 위치를 확인합니다.
        rotate_back = self._normalize_angle(parallel_odom_yaw - self.current_yaw)
        self.get_logger().info(
            f"3단계: 평행 자세로 복귀 회전 {math.degrees(rotate_back):.1f} deg"
        )
        if not self._rotate_relative(rotate_back):
            self._stop_robot()
            return False

        self._sleep_with_spin(self.settle_time)

        retry_total_distance = min(
            line['line_length'] * self.retry_total_move_multiplier,
            self.max_search_distance
        )
        additional_distance = max(0.0, retry_total_distance - first_distance)
        self.get_logger().info(
            f"4단계: 2차 추가 전진 "
            f"(total_multiplier={self.retry_total_move_multiplier:.2f}, "
            f"total_distance={retry_total_distance:.2f} m, "
            f"additional={additional_distance:.2f} m)"
        )

        if additional_distance > self.distance_tolerance:
            if not self._move_straight(additional_distance):
                self._stop_robot()
                return False
            if self.last_move_stopped_by_obstacle:
                self.get_logger().warning(
                    "2차 전진 중 장애물을 만나 현재 위치에서 차량 좌표를 바라봅니다."
                )

        self._stop_robot()
        self._sleep_with_spin(self.settle_time)

        if not self._face_target_pose():
            self._stop_robot()
            return False

        self._sleep_with_spin(self.settle_time)
        if self.retry_yolo_check_sec > 0.0:
            if self._wait_for_plate_detection(
                self.retry_yolo_check_sec,
                "2차 이동 후 번호판 확인"
            ):
                self.get_logger().info("2차 이동 위치에서 번호판 탐지 성공.")
                return True

        self._stop_robot()
        self.get_logger().warning("번호판 최종 탐지 실패.")
        return False

    def _reset_plate_detection_state(self):
        """새 target 확인을 시작하기 전에 이전 번호판 탐지 상태를 지웁니다."""
        self.latest_plate_detected = False
        self.latest_plate_stamp = 0.0
        self.latest_plate_confidence = 0.0
        self.accept_plate_detection = False
        self.accepted_plate_detected = False
        self.accepted_plate_image_msg = None
        self.last_move_stopped_by_obstacle = False

    def _wait_for_plate_recovery_ready(self):
        """라이다 직선, odom, amcl pose가 들어올 때까지 기다립니다."""
        start = time.monotonic()
        self.get_logger().info("라이다 직선, odom, amcl 데이터를 기다립니다...")

        while rclpy.ok() and time.monotonic() - start < self.plate_wait_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            line_is_fresh = (
                self.latest_line is not None
                and time.monotonic() - self.latest_line['stamp'] < 1.0
            )
            odom_ready = self.current_odom is not None and self.current_yaw is not None
            pose_ready = self.current_pose is not None

            if line_is_fresh and odom_ready and pose_ready:
                return True

        self.get_logger().warning("번호판 재탐색에 필요한 센서 데이터를 받지 못했습니다.")
        return False

    def _wait_for_plate_detection(self, duration, label):
        """정지 상태에서 duration 동안 YOLO 번호판 탐지를 성공 판정으로 인정합니다."""
        self._stop_robot()
        self.accepted_plate_detected = False
        self.accepted_plate_image_msg = None
        self.accept_plate_detection = True
        start = time.monotonic()
        self.get_logger().info(f"{label}: {duration:.1f}초 확인")

        while rclpy.ok() and time.monotonic() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.accepted_plate_detected and self.accepted_plate_image_msg is not None:
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
        """현재 AMCL map pose에서 차량 좌표 target_x/y를 바라보도록 제자리 회전합니다."""
        if self.current_pose is None:
            self.get_logger().warning("AMCL pose가 없어 차량 방향을 계산할 수 없습니다.")
            return False

        pose = self.current_pose.pose
        dx = self.target_x - pose.position.x
        dy = self.target_y - pose.position.y

        if math.hypot(dx, dy) < 1e-4:
            self.get_logger().warning("target 좌표가 현재 위치와 너무 가깝습니다.")
            return True

        target_yaw = math.atan2(dy, dx)
        current_map_yaw = self._yaw_from_quaternion(pose.orientation)
        rotate_angle = self._normalize_angle(target_yaw - current_map_yaw)

        self.get_logger().info(
            f"차량 좌표 바라보기: target_yaw={math.degrees(target_yaw):.1f} deg, "
            f"current_yaw={math.degrees(current_map_yaw):.1f} deg, "
            f"rotate={math.degrees(rotate_angle):.1f} deg"
        )
        return self._rotate_relative(rotate_angle)

    def _rotate_relative(self, target_angle):
        """odom yaw 변화를 보면서 target_angle만큼 제자리 회전합니다."""
        if abs(target_angle) <= self.angle_tolerance:
            self.get_logger().info("회전 오차가 작아서 회전을 생략합니다.")
            return True
        if self.current_yaw is None:
            self.get_logger().warning("odom yaw가 없어 회전할 수 없습니다.")
            return False

        start_yaw = self.current_yaw
        direction = 1.0 if target_angle > 0.0 else -1.0
        timeout = max(8.0, abs(target_angle) / max(self.angular_speed, 1e-3) + 8.0)
        start = time.monotonic()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_yaw is None:
                continue

            rotated = self._normalize_angle(self.current_yaw - start_yaw)
            remaining = abs(target_angle) - abs(rotated)

            if remaining <= self.angle_tolerance:
                self._stop_robot()
                return True

            if time.monotonic() - start > timeout:
                self.get_logger().warning("회전 timeout이 발생했습니다.")
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
        if self.current_odom is None:
            self.get_logger().warning("odom 데이터가 없어 직선 이동할 수 없습니다.")
            return False

        self.last_move_stopped_by_obstacle = False
        start_pose = self.current_odom.pose.pose.position
        start_x = start_pose.x
        start_y = start_pose.y
        direction = 1.0 if distance > 0.0 else -1.0
        timeout = max(8.0, abs(distance) / max(self.linear_speed, 1e-3) + 8.0)
        start = time.monotonic()

        self.get_logger().info(f"{distance:+.2f} m 직선 이동")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if direction > 0.0 and self._front_obstacle_too_close():
                self.get_logger().warning("전방 장애물이 가까워 현재 위치에서 이동을 중단합니다.")
                self.last_move_stopped_by_obstacle = True
                self._stop_robot()
                return True

            pose = self.current_odom.pose.pose.position
            moved = math.hypot(pose.x - start_x, pose.y - start_y)
            remaining = abs(distance) - moved

            if remaining <= self.distance_tolerance:
                self._stop_robot()
                return True

            if time.monotonic() - start > timeout:
                self.get_logger().warning("직선 이동 timeout이 발생했습니다.")
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
        """YOLO 결과에서 Plate class만 성공 후보로 보고, bbox가 그려진 이미지를 반환합니다."""
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

        status_text = (
            f"Plate detected: {best_confidence:.2f}"
            if plate_detected
            else "Plate: not detected"
        )
        if not self.accept_plate_detection:
            status_text += " / display only"

        cv2.putText(
            annotated,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if plate_detected else (0, 0, 255),
            2
        )
        return plate_detected, best_confidence, annotated

    def _cv2_to_compressed_msg(self, img, header):
        """bbox가 그려진 OpenCV 이미지를 sensor_msgs/CompressedImage로 변환합니다."""
        ok, encoded = cv2.imencode('.jpg', img)
        if not ok:
            self.get_logger().warning("번호판 이미지 JPEG 인코딩 실패")
            return None

        msg = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        return msg

    def _get_class_label(self, cls):
        """YOLO class id를 표시용 문자열로 변환합니다."""
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, f'class_{cls}')
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return f'class_{cls}'

    def _scan_to_roi_points(self, msg):
        """LaserScan ranges 중 ROI 각도/거리 안의 점만 라이다 2D 좌표로 변환합니다."""
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
        """ROI 점들을 거리 연속성 기준으로 cluster 단위로 나눕니다."""
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
        """여러 cluster 중 로봇과 가장 가까운 cluster를 차량면 후보로 선택합니다."""
        if not clusters:
            return None
        return min(clusters, key=self._cluster_min_distance)

    def _cluster_min_distance(self, cluster):
        """cluster 내부 점들 중 로봇 원점에서 가장 가까운 거리입니다."""
        distances = np.linalg.norm(cluster, axis=1)
        return float(np.min(distances))

    def _fit_line_ransac(self, points):
        """RANSAC으로 이상점을 버리고 직선 후보를 찾습니다."""
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
        """RANSAC inlier를 PCA로 다시 피팅해 최종 직선 방향을 계산합니다."""
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
        """점들과 직선 사이의 수직 거리를 계산합니다."""
        relative = points - point_on_line
        return np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])

    def _line_length(self, point_on_line, direction, points):
        """직선 방향으로 점들을 투영해 라이다에 보이는 차량면 길이를 계산합니다."""
        projections = (points - point_on_line) @ direction
        return float(np.max(projections) - np.min(projections))

    def _publish_plate_markers(self, header, roi_points, selected_cluster, point_on_line, direction, inliers):
        """RViz에서 ROI 점, 선택 cluster, inlier, 최종 초록 선을 확인할 Marker를 발행합니다."""
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
        """numpy 점 배열을 RViz POINTS Marker로 변환합니다."""
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
        """유효한 직선을 못 찾았을 때 RViz에 남은 이전 marker를 지웁니다."""
        for marker_id in [0, 1, 2, 3]:
            marker = Marker()
            marker.header = header
            marker.ns = 'plate_lidar_line'
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)

    def _to_point(self, xy, z=0.0):
        """numpy xy 좌표를 visualization_msgs/Point로 변환합니다."""
        point = Point()
        point.x = float(xy[0])
        point.y = float(xy[1])
        point.z = z
        return point

    def _yaw_from_quaternion(self, q):
        """quaternion orientation에서 yaw만 추출합니다."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_parallel_angle(self, angle):
        """직선 방향 오차를 -90~90도 범위로 정규화합니다."""
        while angle > math.pi / 2.0:
            angle -= math.pi
        while angle < -math.pi / 2.0:
            angle += math.pi
        return angle

    def _normalize_angle(self, angle):
        """일반 yaw 오차를 -180~180도 범위로 정규화합니다."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _stop_robot(self):
        """cmd_vel 0을 발행해 로봇을 정지시킵니다."""
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

    def _sleep_with_spin(self, duration):
        """대기 중에도 scan/odom/image callback이 계속 처리되도록 spin_once를 반복합니다."""
        end_time = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _log_plate_periodically(self, text):
        """라이다 로그가 너무 많이 찍히지 않도록 주기 제한을 둡니다."""
        now = time.monotonic()
        if now - self.last_log_time < self.plate_log_period_sec:
            return
        self.last_log_time = now
        self.get_logger().info(text)

    #-----------
    # Navigation
    #----------- 
    def cancel_task(self):
        if self.current_task:
            self.navigator.cancelTask()
            
            self.current_task = False 

    def follow_path(self):
        if self.wait_until is not None:
            if time.time() < self.wait_until:
                return
            self.wait_until = None

        if self.current_pose is None:
            return
        
        if self.mode == "PATROL":
            paths = self.default_paths
        elif self.mode == "MISSION":
            paths = self.mission_paths
        else:
            paths = self.return_paths
        
        if len(paths) == 0:
            return
        
        if not self.current_task:
            if not self.current_task:

                if self.mode == "PATROL":

                    goal = self.make_pose(
                        "map",
                        PATROL_TARGETS[self.path_index][0],
                        PATROL_TARGETS[self.path_index][1],
                        PATROL_TARGETS[self.path_index][2]
                    )

                    path = self.navigator.getPath(self.current_pose, goal)

                    if path is None:
                        self.get_logger().error("Patrol path generation failed")
                        return

                    self.navigator.followPath(path)

                else:
                    self.navigator.followPath(paths[self.path_index])

                self.current_task = True
                self.get_logger().info(f"start path {self.path_index}")
                return
        
        # 디버깅 용 
        feedback = self.navigator.getFeedback()
        if feedback is not None:
            print(feedback)

        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            self.current_task = False

            if result == TaskResult.SUCCEEDED:
                if self.mode == "MISSION":
                    self.finished_target = self.mission_targets[self.path_index]
                    self.get_logger().info(
                        f"target reached: {self.finished_target['event_id']}, 번호판 확인 시작"
                    )

                    plate_success = self.run_plate_recovery_for_target(self.finished_target)
                    if plate_success:
                        # 번호판 사진을 확보한 경우에만 이미지/id와 기존 /a_to_b를 발행합니다.
                        self.publish_plate_result(self.finished_target)
                        self.publish_target_to_amr2(self.finished_target)
                    else:
                        self.get_logger().warning(
                            f"번호판 확인 실패: event_id={self.finished_target['event_id']} "
                            "전송 없이 다음 좌표로 이동합니다."
                        )

                    self.wait_until = time.time() + 3.0
                self.path_index += 1

                if self.mode == "PATROL":
                    if self.path_index >= len(self.default_paths):
                        self.path_index = 0
                
                elif self.mode == "MISSION":
                    if self.path_index >= len(self.mission_paths):
                        if len(self.pending_targets) > 0:
                            self.get_logger().info("mission complete")
                            self.start_pending_mission()

                        else:
                            self.get_logger().info("all clear go to patrol")
                            self.return_to_patrol()
                
                elif self.mode == "RETURN":
                    if self.path_index >= len(self.return_paths):
                        self.mode = "PATROL"
                        self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)

            elif result == TaskResult.FAILED or result == TaskResult.CANCELED:
                self.get_logger().error(f"{self.path_index} fail")
                self.path_index += 1

                if self.mode == "PATROL" and self.path_index >= len(self.default_paths):
                    self.path_index = 0
                elif self.mode == "MISSION" and self.path_index >= len(self.mission_paths):
                    if len(self.pending_targets) > 0:
                        self.get_logger().info("mission failed start pending")
                        self.start_pending_mission()
                    else:
                        self.return_to_patrol()

                elif self.mode == "RETURN" and self.path_index >= len(self.return_paths):
                    self.mode = "PATROL"
                    self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)

    def destroy_node(self):
        """노드 종료 시 로봇 정지 명령과 OpenCV 창 정리를 수행합니다."""
        self._stop_robot()
        if self.show_yolo_window:
            cv2.destroyAllWindows()
        super().destroy_node()

#-----------
# main
#-----------    
def main(args=None):
    rclpy.init(args=args)

    node = Amrmove()
    try:
        node.get_logger().info("nav 켜지는 중")

        # 나중에 코드에서 Initial Pose를 줄 때 사용
        # node.navigator.waitUntilNav2Active()

        node.get_logger().info("nav 떴다")

        # RViz에서 Initial Pose를 줄 때까지 대기
        node.get_logger().info("/amcl_pose ... loading")

        while rclpy.ok() and node.current_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("amcl_pose received")

        # 순찰 경로 생성
        node.default_path(node.current_pose)

        # 메인 루프
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.follow_path()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 입력으로 amr_move_peter 노드를 종료합니다.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
