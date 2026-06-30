#!/usr/bin/env python3

"""Test publisher that sends one or more AMR1 goals to AMR2."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class Amr1GoalPublisher(Node):
    """Publish configurable fake AMR1 goals."""

    def __init__(self):
        """Initialize the configurable test publisher."""
        super().__init__('test_amr1_goal_publisher')

        self.declare_parameter('goal_topic', '/a_to_b')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter(
            'goals',
            [-4.2, 3.8, 0.0])
        self.declare_parameter('publish_interval_sec', 1.0)
        self.declare_parameter('connection_timeout_sec', 15.0)

        self.goal_topic = str(self.get_parameter('goal_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        raw_goals = list(self.get_parameter('goals').value)
        self.publish_interval_sec = float(
            self.get_parameter('publish_interval_sec').value)
        self.connection_timeout_sec = float(
            self.get_parameter('connection_timeout_sec').value)

        if not raw_goals or len(raw_goals) % 3 != 0:
            raise ValueError(
                'goals는 [x, y, yaw, x, y, yaw, ...] 형식이어야 합니다.')
        if not all(math.isfinite(float(value)) for value in raw_goals):
            raise ValueError('goals에 NaN 또는 무한대가 포함되어 있습니다.')

        self.goals = [
            tuple(float(value) for value in raw_goals[index:index + 3])
            for index in range(0, len(raw_goals), 3)
        ]
        self.next_goal_index = 0
        self.started_ns = self.get_clock().now().nanoseconds
        self.last_publish_ns = 0

        self.publisher = self.create_publisher(
            PoseStamped, self.goal_topic, 10)
        self.timer = self.create_timer(0.1, self._tick)
        self.shutdown_timer = None

        self.get_logger().info(
            f'AMR2 연결 대기: topic={self.goal_topic}, '
            f'goals={len(self.goals)}')

    def _tick(self):
        now_ns = self.get_clock().now().nanoseconds
        elapsed_sec = (now_ns - self.started_ns) / 1_000_000_000

        if self.publisher.get_subscription_count() == 0:
            if elapsed_sec >= self.connection_timeout_sec:
                self.get_logger().error(
                    f'{self.connection_timeout_sec:.1f}초 동안 '
                    f'{self.goal_topic} 구독자를 찾지 못했습니다.')
                self._schedule_shutdown()
            return

        interval_ns = int(self.publish_interval_sec * 1_000_000_000)
        if (
            self.last_publish_ns
            and now_ns - self.last_publish_ns < interval_ns
        ):
            return

        x, y, yaw = self.goals[self.next_goal_index]
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.orientation.w = math.cos(yaw / 2.0)
        self.publisher.publish(message)

        self.next_goal_index += 1
        self.last_publish_ns = now_ns
        self.get_logger().info(
            f'테스트 목표 발행 {self.next_goal_index}/{len(self.goals)}: '
            f'x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')

        if self.next_goal_index >= len(self.goals):
            self._schedule_shutdown()

    def _schedule_shutdown(self):
        self.timer.cancel()
        if self.shutdown_timer is None:
            self.shutdown_timer = self.create_timer(1.0, self._shutdown)

    def _shutdown(self):
        self.shutdown_timer.cancel()
        rclpy.shutdown()


def main(args=None):
    """Run the fake AMR1 goal publisher."""
    rclpy.init(args=args)
    node = None
    try:
        node = Amr1GoalPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'AMR1 test publisher 시작 실패: {error}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
