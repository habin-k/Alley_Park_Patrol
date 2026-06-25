import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

# 다각형 기하학 연산을 위한 라이브러리
# 실행 전 설치 필요: pip install shapely
from shapely.geometry import Point, Polygon

class ZoneDetectorNode(Node):
    def __init__(self):
        super().__init__('zone_detector_node')
        
        # 1. 구역(Zone) 좌표 설정
        # RViz에서 Publish Point로 얻은 맵 상의 (X, Y) 모서리 좌표들을 입력합니다.
        
        # 예시 1: 어린이 보호 구역 (4개의 모서리로 이루어진 사각형 구역)
        school_zone_coords = [(-1.5, 2.0), (1.5, 2.0), (1.5, 4.0), (-1.5, 4.0)]
        self.school_zone = Polygon(school_zone_coords)
        
        # 예시 2: 소방차 전용 구역
        fire_truck_zone_coords = [(3.0, -1.0), (5.0, -1.0), (5.0, 1.0), (3.0, 1.0)]
        self.fire_truck_zone = Polygon(fire_truck_zone_coords)

        # 2. 로봇의 현재 위치(AMCL) 구독
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose', # 로봇의 현재 위치를 알려주는 토픽
            self.pose_callback,
            10)
            
        self.current_zone = "일반 도로"
        self.get_logger().info("✅ 구역 판별 노드가 시작되었습니다.")

    def pose_callback(self, msg):
        # 로봇의 현재 X, Y 좌표 추출
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        
        # Shapely의 Point 객체로 변환
        robot_point = Point(current_x, current_y)

        # 3. 로봇이 어느 구역에 있는지 포함 여부(contains) 판별
        new_zone = "일반 도로"
        
        if self.school_zone.contains(robot_point):
            new_zone = "어린이 보호 구역"
        elif self.fire_truck_zone.contains(robot_point):
            new_zone = "소방차 전용 구역"

        # 구역이 바뀌었을 때만 이벤트 발생
        if new_zone != self.current_zone:
            self.get_logger().info(f"🚨 진입 알림: 로봇이 [{new_zone}]에 진입했습니다! (X:{current_x:.2f}, Y:{current_y:.2f})")
            self.current_zone = new_zone
            
            # 여기서 구역에 따른 후속 작업을 실행합니다.
            self.trigger_zone_action(new_zone)

    def trigger_zone_action(self, zone_name):
        if zone_name == "어린이 보호 구역":
            self.get_logger().info(">>> 속도를 줄이고 어린이 보호 구역 맞춤 AI 탐지 모델을 가동합니다.")
            # 예: 특정 토픽 퍼블리시 또는 서비스 호출
        elif zone_name == "소방차 전용 구역":
            self.get_logger().info(">>> 소방차 구역 불법 주차 집중 단속을 시작합니다.")
        else:
            self.get_logger().info(">>> 일반 주행 모드로 복귀합니다.")

def main(args=None):
    rclpy.init(args=args)
    node = ZoneDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()