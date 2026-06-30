
import json
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


AMR1_GOAL_TOPIC = '/a_to_b'
WEBCAM_DATA_TOPIC = '/webcam_objects/map_detections'


class MissionQueue(Node):
    """AMR1/Webcam 목표를 하나의 mission queue에 저장한다."""

    def __init__(self):
        super().__init__('mission_queue')

        self.mission_queue = deque()

        # AMR1에서 id, x, y가 들어있는 JSON 문자열 수신.
        self.create_subscription(
            String, AMR1_GOAL_TOPIC, self._amr1_callback, 10)

        # Webcam에서 map 좌표가 들어있는 JSON 문자열 수신.
        self.create_subscription(
            String, WEBCAM_DATA_TOPIC, self._webcam_callback, 10)

        self.get_logger().info('Mission queue node ready.')

    def _amr1_callback(self, msg: String):
        """AMR1 JSON을 mission queue 항목으로 변환해서 저장한다."""
        mission = self._parse_amr1_msg(msg)
        if mission is None:
            return

        self._enqueue_mission(mission, 'AMR1')

    def _webcam_callback(self, msg: String):
        """Webcam JSON 안의 detections를 mission queue 항목으로 저장한다."""
        missions = self._parse_webcam_msg(msg)
        for mission in missions:
            self._enqueue_mission(mission, 'Webcam')

    def _parse_amr1_msg(self, msg: String):
        """AMR1 JSON 문자열에서 id, x, y를 꺼낸다."""
        try:
            data = json.loads(msg.data)
            target_id = str(data['id'])

            # {"id": "...", "x": 1.0, "y": 2.0} 형태와
            # {"id": "...", "map": {"x": 1.0, "y": 2.0}} 형태를 모두 허용한다.
            point = data['map'] if 'map' in data else data
            x = self._value_from_keys(point, 'x', 'X')
            y = self._value_from_keys(point, 'y', 'Y')
            orient = self._value_from_keys(point, 'orient', 'orientation', 'yaw', default=0.0)

            return self._make_mission(target_id, x, y, orient)

        except Exception as e:
            self.get_logger().error(f'AMR1 json parse error: {e}')
            return None

    def _parse_webcam_msg(self, msg: String):
        """Webcam JSON 문자열에서 detections[*].map.x/y를 꺼낸다."""
        try:
            data = json.loads(msg.data)
            camera_id = str(data['id'])

            missions = []
            for index, det in enumerate(data['detections']):
                point = det['map']
                x = self._value_from_keys(point, 'x', 'X')
                y = self._value_from_keys(point, 'y', 'Y')
                orient = self._value_from_keys(
                    point, 'orient', 'orientation', 'yaw', default=0.0)

                # 한 카메라 메시지에 detection이 여러 개일 수 있으므로 id를 구분한다.
                target_id = str(det.get('id', f'{camera_id}_{index}'))
                missions.append(self._make_mission(target_id, x, y, orient))

            return missions

        except Exception as e:
            self.get_logger().error(f'Webcam json parse error: {e}')
            return []

    def _make_mission(self, target_id, x, y, orient):
        """큐에 저장할 공통 dict 형태를 만든다."""
        return {
            'id': str(target_id),
            'x': float(x),
            'y': float(y),
            'orient': float(orient),
        }

    def _enqueue_mission(self, mission, source: str):
        """mission queue에 목표를 추가한다."""
        if self._mission_id_exists(mission['id']):
            self.get_logger().info(f'중복 목표 무시: id={mission["id"]}')
            return

        self.mission_queue.append(mission)
        self.get_logger().info(
            f'{source} 목표 저장: id={mission["id"]}, '
            f'x={mission["x"]:.3f}, y={mission["y"]:.3f}, '
            f'orient={mission["orient"]:.3f}, '
            f'queue={len(self.mission_queue)}')

    def _mission_id_exists(self, target_id: str) -> bool:
        """이미 큐에 같은 id가 있는지 확인한다."""
        return any(mission['id'] == target_id for mission in self.mission_queue)

    def _value_from_keys(self, data, *keys, default=None):
        """여러 후보 key 중 존재하는 값을 반환한다."""
        for key in keys:
            if key in data:
                return data[key]
        if default is not None:
            return default
        raise KeyError('/'.join(keys))


def main(args=None):
    """ROS2 node entry point."""
    rclpy.init(args=args)
    node = MissionQueue()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
