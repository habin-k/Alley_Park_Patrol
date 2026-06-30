#!/usr/bin/env python3

"""Request-driven fake OCR node for AMR2 testing."""

import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import String


class FakeOcrNode(Node):
    """Return a configurable Boolean result for each OCR request."""

    def __init__(self):
        """Initialize OCR request and result topics."""
        super().__init__('test_fake_ocr_node')

        self.declare_parameter('image_topic', '/target_plate_image')
        self.declare_parameter('id_topic', '/plate_id')
        self.declare_parameter('result_topic', '/ocr_result')
        self.declare_parameter('result', True)
        self.declare_parameter('alternate_result', False)
        self.declare_parameter('response_delay_sec', 1.0)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.id_topic = str(self.get_parameter('id_topic').value)
        self.result_topic = str(self.get_parameter('result_topic').value)
        self.result = bool(self.get_parameter('result').value)
        self.alternate_result = bool(
            self.get_parameter('alternate_result').value)
        self.response_delay_sec = float(
            self.get_parameter('response_delay_sec').value)

        if self.response_delay_sec < 0.0:
            raise ValueError('response_delay_sec는 0 이상이어야 합니다.')

        self.result_pub = self.create_publisher(
            Bool, self.result_topic, 10)
        self.create_subscription(
            Image, self.image_topic, self._image_callback, 10)
        self.create_subscription(
            String, self.id_topic, self._id_callback, 10)
        self.pending_timers = []
        self.latest_plate_id = None

        self.get_logger().info(
            f'가짜 OCR 준비: image={self.image_topic}, id={self.id_topic}, '
            f'result={self.result_topic}, value={self.result}')

    def _id_callback(self, msg: String):
        self.latest_plate_id = msg.data
        try:
            target = json.loads(msg.data)
            self.get_logger().info(
                f'target 수신: event_id={target.get("event_id")}, '
                f'zone={target.get("zone")}')
        except json.JSONDecodeError:
            self.get_logger().info(f'target 수신: {self.latest_plate_id}')

    def _image_callback(self, msg: Image):
        self.get_logger().info(
            f'OCR 요청 수신. {self.response_delay_sec:.1f}초 후 응답합니다.')
        timer = self.create_timer(
            max(self.response_delay_sec, 0.001),
            lambda: self._publish_result(timer))
        self.pending_timers.append(timer)

    def _publish_result(self, timer):
        timer.cancel()
        if timer in self.pending_timers:
            self.pending_timers.remove(timer)

        message = Bool()
        message.data = self.result
        self.result_pub.publish(message)
        self.get_logger().info(f'가짜 OCR 결과 발행: {message.data}')

        if self.alternate_result:
            self.result = not self.result


def main(args=None):
    """Run the fake OCR node."""
    rclpy.init(args=args)
    node = None
    try:
        node = FakeOcrNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'Fake OCR 시작 실패: {error}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
