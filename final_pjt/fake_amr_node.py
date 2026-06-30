import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher_node')
        # 'a_to_b' 토픽으로 퍼블리셔 생성
        self.publisher_ = self.create_publisher(PoseStamped, 'a_to_b', 10)
        
        # 노드 실행 후 통신이 연결될 시간을 벌기 위해 2초 후 퍼블리시
        self.timer = self.create_timer(2.0, self.publish_goal)
        self.get_logger().info('Goal Publisher가 시작되었습니다. 2초 후 목표를 전송합니다.')

    def publish_goal(self):
        msg = PoseStamped()
        
        # 헤더 설정 (현재 시간 및 프레임 지정)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # Nav2 기본 글로벌 프레임
        
        # 이동할 목표 좌표 (원하는 위치로 수정하세요)
        msg.pose.position.x = 2.0
        msg.pose.position.y = 2.0
        msg.pose.position.z = 0.0
        
        # 방향 설정 (Quaternion, 회전 없음 = w: 1.0)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'목표 좌표 전송 완료: x={msg.pose.position.x}, y={msg.pose.position.y}')
        
        # 목표 좌표를 한 번만 전송하기 위해 타이머 정지
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()