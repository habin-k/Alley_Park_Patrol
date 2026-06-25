# plate_faill_logic.py

# AMR이 좌표값을 받고 이동했을 때 카메라에서 번호판이 보이지 않으면
# YOLO로 번호판을 다시 탐지하기 위해 로봇을 보정 이동시키는 테스트 코드입니다.

import math
import os
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO


class PlateFailTestNode(Node):
    def __init__(self, model, camera_topic='/robot2/oakd/rgb/image_raw/compressed'):
        super().__init__('plate_fail_test_node')

        self.plate_detected = False
        self.target_info_published = False
        self.plate_image_data = None
        self.plate_coords_data = None

        self.current_odom = None
        self.current_yaw = None

        self.model = model
        self.class_names = model.names
        self.bridge = CvBridge()
        self.plate_class_id = 2

        self.declare_parameter('cmd_vel_topic', '/robot2/cmd_vel')
        self.declare_parameter('odom_topic', '/robot2/odom')
        self.declare_parameter('plate_confidence_threshold', 0.5)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.plate_confidence_threshold = (
            self.get_parameter('plate_confidence_threshold').value
        )

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.target_image_pub = self.create_publisher(Image, '/target_plate_image', 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, '/current_pose', 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_callback,
            10
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.process_frame,
            10
        )

        self.get_logger().info(
            f"번호판 재탐색 테스트 노드 시작 "
            f"(cmd_vel: {cmd_vel_topic}, odom: {odom_topic}, camera: {camera_topic})"
        )
        self.get_logger().info(f"YOLO class names: {self.class_names}")

    # ---------------------------------------------------------
    # YOLO 이미지 처리 및 타겟 정보 전송
    # ---------------------------------------------------------
    def process_frame(self, msg):
        if self.target_info_published:
            return

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"compressed image 변환 실패: {exc}")
            return

        results = self.model(img, verbose=False)
        best_detection = self._find_best_plate_detection(results)

        if best_detection is None:
            self.plate_detected = False
            return

        x1, y1, x2, y2, confidence, label = best_detection
        self.plate_detected = True
        self.plate_coords_data = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': confidence,
            'label': label,
        }

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"{label}: {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header = msg.header
        self.plate_image_data = image_msg

        self._stop_robot()
        self.send_target_info_to_amr2()

    def _find_best_plate_detection(self, results):
        best_detection = None
        best_confidence = -1.0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.plate_confidence_threshold:
                    continue

                cls = int(box.cls[0])
                if cls != self.plate_class_id:
                    continue

                label = self._get_class_label(cls)
                if confidence > best_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_confidence = confidence
                    best_detection = (x1, y1, x2, y2, confidence, label)

        return best_detection

    def _get_class_label(self, cls):
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, f'class_{cls}')
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return f'class_{cls}'

    def send_target_info_to_amr2(self):
        if self.target_info_published:
            return True

        if self.plate_image_data is None:
            self.get_logger().warn("전송할 번호판 이미지가 아직 없습니다.")
            return False

        if self.current_odom is None:
            self.get_logger().warn("현재 odom 위치가 아직 없습니다. 위치 전송을 보류합니다.")
            return False

        pose_msg = PoseStamped()
        pose_msg.header = self.current_odom.header
        pose_msg.header.frame_id = self.current_odom.header.frame_id or 'odom'
        pose_msg.pose = self.current_odom.pose.pose

        self.target_image_pub.publish(self.plate_image_data)
        self.target_pose_pub.publish(pose_msg)
        self.target_info_published = True

        coords = self.plate_coords_data
        self.get_logger().info(
            f"번호판 이미지와 현재 위치 전송 완료: "
            f"label={coords['label']}, confidence={coords['confidence']:.2f}, "
            f"bbox=({coords['x1']}, {coords['y1']}, {coords['x2']}, {coords['y2']})"
        )
        return True

    def resume_patrol_loop(self):
        self.get_logger().info("지정된 순찰 루트(Patrol Loop)로 복귀하여 다음 지점으로 이동합니다.")

    # ---------------------------------------------------------
    # 번호판 미인식 시 재탐색 시퀀스
    # ---------------------------------------------------------
    def execute_recovery_routine(self):
        self.get_logger().info("번호판 미인식: 재탐색 보정 로직을 시작합니다.")

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f"재탐색 시도 {attempt}/{max_retries}")

            if self._wait_and_check(3.0):
                return True

            self.get_logger().info("- 2단계: 45도 좌우 스캔")
            if self._scan_left_right():
                return True

            self.get_logger().info("- 3단계: 180도 회전 후 15cm 전진, 다시 180도 원복")
            if self._rotate_in_place(180):
                return True
            if self._move_straight(0.15):
                return True
            if self._rotate_in_place(180):
                return True
            if self._scan_left_right():
                return True

            self.get_logger().info("- 4단계: 90도 측면 이동 후 차를 바라봄")
            if self._rotate_in_place(90):
                return True
            if self._move_straight(0.20):
                return True
            if self._rotate_in_place(-90):
                return True
            if self._wait_and_check(2.0):
                return True

        self.get_logger().warn("3회 재탐색 실패: 번호판을 찾을 수 없습니다.")
        return False

    # ---------------------------------------------------------
    # odom 기반 보정 이동
    # ---------------------------------------------------------
    def _odom_callback(self, msg):
        self.current_odom = msg
        self.current_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)

    def _quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _spin_once_and_check(self, timeout_sec=0.1):
        rclpy.spin_once(self, timeout_sec=timeout_sec)
        if self.plate_detected and not self.target_info_published:
            self.send_target_info_to_amr2()

        if self.plate_detected and self.target_info_published:
            self._stop_robot()
            self.get_logger().info("번호판 인식 및 전송 성공. 로봇 정지.")
            return True
        return False

    def _wait_for_odom(self, timeout_sec=3.0):
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and self.current_odom is None and time.monotonic() < end_time:
            if self._spin_once_and_check(0.1):
                return True

        if self.current_odom is None:
            self.get_logger().error("odom 데이터를 받지 못했습니다. 로봇 보정 이동을 건너뜁니다.")
            return False
        return True

    def _wait_and_check(self, duration_sec):
        steps = int(duration_sec * 10)
        for _ in range(steps):
            if self._spin_once_and_check(0.1):
                return True
        return False

    def _scan_left_right(self):
        if self._rotate_in_place(45):
            return True
        if self._rotate_in_place(-90):
            return True
        if self._rotate_in_place(45):
            return True
        return False

    def _rotate_in_place(self, degrees):
        if degrees == 0:
            return self._spin_once_and_check(0.1)

        if self.target_info_published:
            self._stop_robot()
            return True

        if not self._wait_for_odom():
            return False

        if self.target_info_published:
            self._stop_robot()
            return True

        target_radians = math.radians(abs(degrees))
        direction = 1.0 if degrees > 0 else -1.0
        angular_speed = 0.3
        angle_tolerance = math.radians(2.0)
        timeout_sec = target_radians / angular_speed + 3.0

        twist = Twist()
        twist.angular.z = direction * angular_speed

        prev_yaw = self.current_yaw
        rotated = 0.0
        start_time = time.monotonic()

        while rclpy.ok() and abs(rotated) < target_radians - angle_tolerance:
            if time.monotonic() - start_time > timeout_sec:
                self.get_logger().warn(f"{degrees}도 회전 제한 시간 초과")
                break

            self.cmd_pub.publish(twist)
            if self._spin_once_and_check(0.1):
                return True

            if self.current_yaw is None:
                continue

            delta_yaw = self._normalize_angle(self.current_yaw - prev_yaw)
            rotated += delta_yaw
            prev_yaw = self.current_yaw

            if rotated * direction < -angle_tolerance:
                self.get_logger().warn("odom 회전 방향이 명령 방향과 다릅니다. 회전을 중단합니다.")
                break

        self._stop_robot()
        return False

    def _move_straight(self, distance_m):
        if distance_m == 0:
            return self._spin_once_and_check(0.1)

        if self.target_info_published:
            self._stop_robot()
            return True

        if not self._wait_for_odom():
            return False

        if self.target_info_published:
            self._stop_robot()
            return True

        linear_speed = 0.1
        direction = 1.0 if distance_m > 0 else -1.0
        target_distance = abs(distance_m)
        distance_tolerance = 0.02
        timeout_sec = target_distance / linear_speed + 3.0

        twist = Twist()
        twist.linear.x = direction * linear_speed

        start_position = self.current_odom.pose.pose.position
        start_x = start_position.x
        start_y = start_position.y
        start_time = time.monotonic()

        while rclpy.ok():
            position = self.current_odom.pose.pose.position
            moved_distance = math.hypot(position.x - start_x, position.y - start_y)

            if moved_distance >= target_distance - distance_tolerance:
                break

            if time.monotonic() - start_time > timeout_sec:
                self.get_logger().warn(f"{distance_m:.2f}m 직진 제한 시간 초과")
                break

            self.cmd_pub.publish(twist)
            if self._spin_once_and_check(0.1):
                return True

        self._stop_robot()
        return False

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())


def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        return YOLO(model_path)
    if suffix in ['.onnx', '.engine']:
        return YOLO(model_path, task='detect')
    raise ValueError(f"지원하지 않는 모델 형식입니다: {suffix}")


def main(args=None):
    rclpy.init(args=args)

    model_path = '/home/rokey/rokey_ws/src/final_pjt/final_pjt/semi_allimages_v5n.pt'

    try:
        model = load_yolo_model(model_path)
        node = PlateFailTestNode(model)
    except Exception as exc:
        print(exc)
        if rclpy.ok():
            rclpy.shutdown()
        return

    node.get_logger().info(">>> 불법주차 의심 차량 목적지에 도착했습니다. YOLO 탐색 시작...")

    try:
        is_success = node.execute_recovery_routine()

        if is_success:
            node.resume_patrol_loop()
        else:
            node.get_logger().warn("인식 불가로 해당 차량 확인을 종료합니다.")
            node.resume_patrol_loop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
