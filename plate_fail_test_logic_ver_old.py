# plate_fail_test_logic.py

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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO
from std_msgs.msg import Bool, String

class PlateFailTestNode(Node):
    """번호판 미탐지 상황에서 YOLO 재탐지와 로봇 보정 이동을 테스트하는 노드."""

    def __init__(self, model, camera_topic='/robot2/oakd/rgb/image_raw/compressed'):
        super().__init__('plate_fail_test_node')
     

        # 번호판 탐지 및 전송 상태 플래그입니다.
        # plate_detected: 현재 YOLO 프레임에서 번호판을 찾았는지 여부
        # target_info_published: 번호판 이미지와 로봇 위치를 이미 전송했는지 여부
        self.plate_detected = False
        self.target_info_published = False

        # 번호판을 찾았을 때 publish할 이미지와 bbox 정보를 임시 저장합니다.
        self.plate_image_data = None
        self.plate_coords_data = None

        # odom 콜백에서 계속 갱신되는 로봇 현재 위치와 yaw입니다.
        self.current_odom = None
        self.current_yaw = None

        # YOLO 모델, 클래스 이름, OpenCV/ROS 이미지 변환기입니다.
        self.model = model
        self.class_names = model.names
        self.bridge = CvBridge()
        self.plate_class_id = 2  # YOLO class id 2번이 Plate입니다.
        self.window_name = 'plate_fail_yolo_view'  # 카메라 화면 확인용 OpenCV 창 이름입니다.

        # 토픽 이름과 threshold를 실행 시 --ros-args -p 로 바꿀 수 있게 parameter로 둡니다.
        self.declare_parameter('cmd_vel_topic', '/robot2/cmd_vel')
        self.declare_parameter('odom_topic', '/robot2/odom')
        self.declare_parameter('plate_confidence_threshold', 0.80)  # 번호판 confidence 기준입니다.
        self.declare_parameter('show_yolo_window', True)  # OpenCV 창 표시 여부입니다.

        # 위에서 선언한 parameter 값을 실제 변수로 가져옵니다.
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.plate_confidence_threshold = (
            self.get_parameter('plate_confidence_threshold').value
        )
        self.show_yolo_window = self.get_parameter('show_yolo_window').value

        # 전송 결과 토픽은 한 번 publish한 뒤 echo를 늦게 켜도 볼 수 있도록 transient_local로 둡니다.
        target_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # 로봇 보정 이동을 위한 속도 명령 publisher입니다.
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # 번호판 탐지 성공 시 ocr 노드로 보낼 이미지와 amr2로 보낼 현재 pose publisher입니다.
        self.target_image_pub = self.create_publisher(
            Image,
            '/target_plate_image',
            target_info_qos
        )
        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/current_pose',
            target_info_qos
        )

        self.plate_id = self.create_publisher(
            String,
            '/plate_id',
            10
        )

        # odom은 보정 이동의 실제 회전량/이동거리를 확인하는 데 사용합니다.
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_callback,
            10
        )

        # AMR 카메라 compressed image를 받아 YOLO 추론을 수행합니다.
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
        """카메라 프레임을 받아 YOLO 추론 후 번호판 탐지 결과를 화면과 토픽으로 보냅니다."""
        try:
            # CompressedImage 메시지를 OpenCV BGR 이미지로 변환합니다.
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"compressed image 변환 실패: {exc}")
            return

        # YOLO 추론을 실행하고, 여러 결과 중 가장 confidence가 높은 Plate만 선택합니다.
        results = self.model(img, verbose=False)
        best_detection = self._find_best_plate_detection(results)

        if best_detection is None:
            # 번호판이 보이지 않는 프레임이면 상태를 False로 두고 화면에 표시만 합니다.
            self.plate_detected = False
            cv2.putText(
                img,
                'Plate: not detected',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            self._show_yolo_window(img)
            return

        # 번호판이 탐지되면 bbox, confidence, label을 저장합니다.
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

        # 확인용 OpenCV 화면에 번호판 bbox와 confidence를 표시합니다.
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
        self._show_yolo_window(img)

        # 토픽으로 보낼 수 있도록 OpenCV 이미지를 sensor_msgs/Image로 변환합니다.
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header = msg.header
        self.plate_image_data = image_msg

        # 최초 탐지 시 한 번만 로봇을 멈추고 이미지/위치 정보를 전송합니다.
        if not self.target_info_published:
            self._stop_robot()
            self.send_target_info_to_amr2()

    def _find_best_plate_detection(self, results):
        """YOLO 결과에서 Plate class만 골라 가장 confidence가 높은 bbox를 반환합니다."""
        best_detection = None
        best_confidence = -1.0

        for result in results:
            for box in result.boxes:
                # threshold보다 낮은 탐지는 무시합니다.
                confidence = float(box.conf[0])
                if confidence < self.plate_confidence_threshold:
                    continue

                # 현재 모델에서 class id 2번만 Plate로 사용합니다.
                cls = int(box.cls[0])
                if cls != self.plate_class_id:
                    continue

                # 같은 프레임에 Plate가 여러 개 있으면 confidence가 가장 높은 것만 사용합니다.
                label = self._get_class_label(cls)
                if confidence > best_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_confidence = confidence
                    best_detection = (x1, y1, x2, y2, confidence, label)

        return best_detection

    def _get_class_label(self, cls):
        """YOLO class id를 사람이 읽을 수 있는 label 문자열로 변환합니다."""
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, f'class_{cls}')
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return f'class_{cls}'

    def _show_yolo_window(self, img):
        """디버깅용 OpenCV 창에 현재 YOLO 결과 이미지를 표시합니다."""
        if not self.show_yolo_window:
            return

        try:
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().warn(f"YOLO 확인 창 표시 실패: {exc}")
            self.show_yolo_window = False

    def send_target_info_to_amr2(self):
        """번호판 이미지와 현재 odom pose를 각각 /target_plate_image, /current_pose로 전송합니다."""
        if self.target_info_published:
            return True

        if self.plate_image_data is None:
            self.get_logger().warn("전송할 번호판 이미지가 아직 없습니다.")
            return False

        if self.current_odom is None:
            self.get_logger().warn("현재 odom 위치가 아직 없습니다. 위치 전송을 보류합니다.")
            return False

        # Odometry의 pose 부분만 PoseStamped 형태로 복사해 전송합니다.
        pose_msg = PoseStamped()
        pose_msg.header = self.current_odom.header
        pose_msg.header.frame_id = self.current_odom.header.frame_id or 'odom'
        pose_msg.pose = self.current_odom.pose.pose

        # 번호판 이미지와 현재 위치를 publish하고, 중복 전송 방지 플래그를 세웁니다.
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
        """실제 순찰 복귀 로직 자리입니다. 현재 테스트 코드에서는 로그만 출력합니다."""
        self.get_logger().info("지정된 순찰 루트(Patrol Loop)로 복귀하여 다음 지점으로 이동합니다.")

    def keep_alive_for_test(self):
        """번호판 인식 후에도 창과 노드를 유지해 테스트 결과를 확인할 수 있게 합니다."""
        self.get_logger().info("테스트 확인 모드 유지 중입니다. 종료하려면 Ctrl+C를 누르세요.")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.show_yolo_window:
                cv2.waitKey(1)

    # ---------------------------------------------------------
    # 번호판 미인식 시 재탐색 시퀀스
    # ---------------------------------------------------------
    def execute_recovery_routine(self):
        """번호판이 보이지 않을 때 로봇을 조금씩 움직이며 카메라 시야를 다시 맞춥니다."""
        self.get_logger().info("번호판 미인식: 재탐색 보정 로직을 시작합니다.")

        # 전체 재탐색 시퀀스를 몇 번 반복할지 정합니다.
        max_retries = 2  # 최대 횟수 조절

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f"재탐색 시도 {attempt}/{max_retries}")

            # 1단계: 제자리에서 잠시 기다리며 YOLO가 번호판을 잡는지 확인합니다.
            if self._wait_and_check(3.0):
                return True

            # 2단계: 좌우로 카메라 방향을 흔들어 번호판이 화면에 들어오는지 확인합니다.
            self.get_logger().info("- 2단계: 45도 좌우 스캔")
            if self._scan_left_right():
                return True

            # 3단계: 뒤쪽/앞쪽 위치를 바꿔 다른 시야에서 다시 확인합니다.
            self.get_logger().info("- 3단계: 5cm 후진 후 다시 좌우 스캔")
            if self._move_straight(-0.05):
                return True
            if self._settle_after_motion(0.5):
                return True
            if self._scan_left_right():
                return True

        self.get_logger().warn("3회 재탐색 실패: 번호판을 찾을 수 없습니다.")
        return False

    # ---------------------------------------------------------
    # odom 기반 보정 이동
    # ---------------------------------------------------------
    def _odom_callback(self, msg):
        """odom 메시지를 받을 때마다 현재 pose와 yaw를 갱신합니다."""
        self.current_odom = msg
        self.current_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)

    def _quaternion_to_yaw(self, q):
        """odom orientation quaternion에서 yaw(z축 회전각)만 계산합니다."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """각도를 -pi ~ pi 범위로 정규화해 yaw wrap-around를 처리합니다."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _spin_once_and_check(self, timeout_sec=0.1):
        """ROS 콜백을 한 번 처리하고 번호판 전송 완료 여부를 확인합니다."""
        rclpy.spin_once(self, timeout_sec=timeout_sec)

        # 번호판은 잡혔지만 odom이 늦게 들어와 전송이 보류됐던 경우를 다시 시도합니다.
        if self.plate_detected and not self.target_info_published:
            self.send_target_info_to_amr2()

        # 이미지와 위치 전송이 모두 끝났으면 recovery를 성공으로 종료합니다.
        if self.plate_detected and self.target_info_published:
            self._stop_robot()
            self.get_logger().info("번호판 인식 및 전송 성공. 로봇 정지.")
            return True
        return False

    def _wait_for_odom(self, timeout_sec=3.0):
        """보정 이동 전에 odom 데이터가 들어올 때까지 잠시 기다립니다."""
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and self.current_odom is None and time.monotonic() < end_time:
            if self._spin_once_and_check(0.1):
                return True

        if self.current_odom is None:
            self.get_logger().error("odom 데이터를 받지 못했습니다. 로봇 보정 이동을 건너뜁니다.")
            return False
        return True

    def _wait_and_check(self, duration_sec):
        """duration_sec 동안 정지 상태로 기다리며 YOLO 탐지 성공 여부를 확인합니다."""
        steps = int(duration_sec * 10)
        for _ in range(steps):
            if self._spin_once_and_check(0.1):
                return True
        return False

    def _scan_left_right(self):
        """왼쪽 45도, 오른쪽 90도, 다시 왼쪽 45도 회전해 원래 방향으로 돌아옵니다."""
        if self._rotate_in_place(45):
            return True
        if self._settle_after_motion(0.5):
            return True

        if self._rotate_in_place(-90):
            return True
        if self._settle_after_motion(0.5):
            return True

        if self._rotate_in_place(45):
            return True
        return False

    def _settle_after_motion(self, duration_sec):
        """방향 전환 직후 관성/odom 지연을 줄이기 위해 잠시 정지합니다."""
        self._stop_robot()
        return self._wait_and_check(duration_sec)

    def _rotate_in_place(self, degrees):
        """odom yaw 변화량을 기준으로 제자리 회전합니다. 성공 반환은 번호판 탐지 성공을 의미합니다."""
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
        angular_speed = 0.2  # rad/s
        angle_tolerance = math.radians(2.0)  # 목표 각도에 이 정도 가까워지면 완료로 봅니다.
        timeout_sec = target_radians / angular_speed + 3.0  # odom 지연을 고려한 안전 시간입니다.

        twist = Twist()
        twist.angular.z = direction * angular_speed

        # yaw가 -pi/pi를 넘나들어도 누적 회전량을 계산할 수 있도록 이전 yaw와 차이를 더합니다.
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

            # 이번 루프에서 실제로 회전한 yaw 변화량을 누적합니다.
            delta_yaw = self._normalize_angle(self.current_yaw - prev_yaw)
            rotated += delta_yaw
            prev_yaw = self.current_yaw

        self._stop_robot()
        return False

    def _move_straight(self, distance_m):
        """odom 위치 변화량을 기준으로 직진합니다. 성공 반환은 번호판 탐지 성공을 의미합니다."""
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

        linear_speed = 0.1  # m/s
        direction = 1.0 if distance_m > 0 else -1.0
        target_distance = abs(distance_m)
        distance_tolerance = 0.02  # 2cm 이내로 가까워지면 완료로 봅니다.
        timeout_sec = target_distance / linear_speed + 3.0  # odom 지연을 고려한 안전 시간입니다.

        twist = Twist()
        twist.linear.x = direction * linear_speed

        # 시작 위치를 저장하고, 현재 위치와의 직선 거리를 계산해 목표 거리 도달 여부를 판단합니다.
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
        """linear/angular 속도가 모두 0인 Twist를 보내 로봇을 정지시킵니다."""
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        """노드 종료 시 OpenCV 창도 같이 닫습니다."""
        if self.show_yolo_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def load_yolo_model(model_path):
    """모델 파일 확장자에 맞춰 Ultralytics YOLO 모델을 로드합니다."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        return YOLO(model_path)
    if suffix in ['.onnx', '.engine']:
        return YOLO(model_path, task='detect')
    raise ValueError(f"지원하지 않는 모델 형식입니다: {suffix}")


def main(args=None):
    """ROS2 노드를 초기화하고 번호판 재탐색 테스트 시퀀스를 실행합니다."""
    rclpy.init(args=args)

    # 테스트에 사용할 번호판/차량 탐지 모델 경로입니다.
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
        # 실제 재탐색 로직을 한 번 실행합니다.
        is_success = node.execute_recovery_routine()

        if is_success:
            node.resume_patrol_loop()
        else:
            node.get_logger().warn("인식 불가로 해당 차량 확인을 종료합니다.")
            node.resume_patrol_loop()

        # 테스트에서는 결과 확인을 위해 Ctrl+C 전까지 노드를 유지합니다.
        node.keep_alive_for_test()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 입력으로 테스트 노드를 종료합니다.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
