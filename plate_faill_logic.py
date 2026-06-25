# plate_faill_logic.py


# amr 이 좌표값을 받고 그 좌표값으로 이동했을 때 amr의 카메라에서 번호판이 안 보이면 (yolo로 탐지 했을 때 번호판 이 탐지가 안 되면) 구동하는 로직을
# 테스트 하는 코드 
# amr 구동 노드에 포함시킬 예정, 따로 노드로는 안 만듦

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
import time
import math
import threading # (필요 시) YOLO 처리를 백그라운드로 돌릴 때 사용

class AMRControlNode(Node):
    def __init__(self):
        super().__init__('amr_control_node')
        
        # [내부 변수] YOLO가 번호판을 찾았는지 여부 (토픽 구독 안 함)
        # 같은 스크립트 내의 YOLO 추론 함수에서 이 변수를 True/False로 업데이트해야 합니다.
        self.plate_detected = False 
        self.plate_image_data = None # 찾았을 때의 이미지 데이터를 저장할 변수
        self.plate_coords_data = None # 찾았을 때의 바운딩 박스 정보 등을 저장할 변수

        # [발행] 로봇의 바퀴를 제어하는 토픽 (robot2 에서 테스트 할 예정)
        self.cmd_pub = self.create_publisher(Twist, '/robot2/cmd_vel', 10)
        
        # [발행] AMR2(또는 관제 서버)로 보낼 타겟 정보 토픽
        self.target_image_pub = self.create_publisher(Image, '/target_plate_image', 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, '/current_pose', 10)
        
        self.get_logger().info(" 통합 AMR 제어 노드(YOLO 포함) 시작됨")

        # (예시) 실제 환경에서는 이 노드 안 어딘가에 카메라 이미지를 받아서
        # YOLO 모델에 넣고 돌리는 콜백(또는 루프)이 존재할 것입니다.
        # def image_callback(self, msg):
        #     result = self.yolo_model.predict(msg)
        #     if result.has_plate():
        #         self.plate_detected = True
        #         self.plate_image_data = ...
        #     else:
        #         self.plate_detected = False

    # ---------------------------------------------------------
    # [핵심 로직 1] 번호판 미인식 시 재탐색(Recovery) 시퀀스 함수
    # ---------------------------------------------------------
    def execute_recovery_routine(self):
        self.get_logger().info(" 번호판 미인식: 재탐색 보정 로직을 시작합니다.")
        
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f" 재탐색 시도 {attempt}/{max_retries}")

            if self._wait_and_check(3.0): return True

            self.get_logger().info(" - 2단계: 45도 좌우 도리도리 스캔")
            if self._scan_left_right(): return True

            self.get_logger().info(" - 3단계: 180도 회전 후 15cm 전진, 다시 180도 원복")
            if self._rotate_in_place(180): return True
            if self._move_straight(0.15): return True
            if self._rotate_in_place(180): return True
            if self._scan_left_right(): return True

            self.get_logger().info(" - 4단계: 90도 측면 이동 후 차를 바라봄")
            if self._rotate_in_place(90): return True
            if self._move_straight(0.20): return True
            if self._rotate_in_place(-90): return True
            if self._wait_and_check(2.0): return True

        self.get_logger().warn(" 3회 재탐색 실패: 번호판을 찾을 수 없습니다.")
        return False

    # ---------------------------------------------------------
    # [핵심 로직 2] 타겟 정보 전송 및 순찰 복귀 로직
    # ---------------------------------------------------------
    def send_target_info_to_amr2(self):
        self.get_logger().info(" AMR2(및 서버)로 차량 이미지와 현재 위치 토픽을 전송합니다.")
        # self.plate_image_data 에 저장된 이미지를 전송
        time.sleep(1) # 전송 대기 시간 (시뮬레이션)
        self.get_logger().info(" 전송 완료!")

    def resume_patrol_loop(self):
        self.get_logger().info(" 지정된 순찰 루트(Patrol Loop)로 복귀하여 다음 지점으로 이동합니다...")

    # ---------------------------------------------------------
    # 보조 제어 함수들 (내부 변수 self.plate_detected 참조)
    # ---------------------------------------------------------
    def _wait_and_check(self, duration_sec):
        steps = int(duration_sec * 10)
        for _ in range(steps):
            # 토픽 구독 없이, 같은 클래스 내의 변수를 바로 확인
            if self.plate_detected:
                self._stop_robot()
                self.get_logger().info(" 번호판 인식 성공! 로봇 정지.")
                return True
            time.sleep(0.1)
        return False

    def _scan_left_right(self):
        if self._rotate_in_place(45): return True
        if self._rotate_in_place(-90): return True
        if self._rotate_in_place(45): return True
        return False

    def _rotate_in_place(self, degrees):
        radians = math.radians(degrees)
        angular_speed = 0.3
        duration = abs(radians / angular_speed)
        
        twist = Twist()
        twist.angular.z = angular_speed if degrees > 0 else -angular_speed
        
        steps = int(duration * 10)
        for _ in range(steps):
            if self.plate_detected:
                self._stop_robot()
                return True
            self.cmd_pub.publish(twist)
            time.sleep(0.1)
            
        self._stop_robot()
        return False

    def _move_straight(self, distance_m):
        linear_speed = 0.1
        duration = distance_m / linear_speed
        
        twist = Twist()
        twist.linear.x = linear_speed
        
        steps = int(duration * 10)
        for _ in range(steps):
            if self.plate_detected:
                self._stop_robot()
                return True
            self.cmd_pub.publish(twist)
            time.sleep(0.1)
            
        self._stop_robot()
        return False

    def _stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)


# 테스트용 실행 로직
def main(args=None):
    rclpy.init(args=args)
    node = AMRControlNode()
    
    node.get_logger().info(">>> 불법주차 의심 차량 목적지에 도착했습니다. YOLO 탐색 시작...")
    
    # 1. 번호판 즉시 인식 실패 시 복구 로직 가동
    if not node.plate_detected:
        is_success = node.execute_recovery_routine()
    else:
        node.get_logger().info("도착하자마자 번호판 발견!")
        is_success = True

    # 2. 결과에 따른 후속 처리 및 순찰 복귀
    if is_success:
        node.send_target_info_to_amr2()
        node.resume_patrol_loop()
    else:
        node.get_logger().warn("인식 불가로 해당 차량 확인을 종료합니다.")
        node.resume_patrol_loop()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()