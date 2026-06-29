# yolo_test.py

import os
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO


class YoloTestNode(Node):
    def __init__(self):
        super().__init__('yolo_test_node')

        self.declare_parameter(
            'model_path',
            '/home/rokey/rokey_ws/src/final_pjt/final_pjt_peter/semi_allimages_v5n.pt'
        )
        self.declare_parameter('camera_topic', '/robot4/oakd/rgb/image_raw/compressed')
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('plate_class_id', 2)
        self.declare_parameter('show_all_classes', False)

        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.plate_class_id = self.get_parameter('plate_class_id').value
        self.show_all_classes = self.get_parameter('show_all_classes').value

        self.model = self._load_yolo_model(model_path)
        self.class_names = self.model.names
        self.bridge = CvBridge()
        self.window_name = 'yolo_test_view'

        self.image_sub = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.process_frame,
            10
        )

        self.get_logger().info(
            f"YOLO 테스트 노드 시작 "
            f"(model: {model_path}, camera: {camera_topic})"
        )
        self.get_logger().info(f"YOLO class names: {self.class_names}")

    def _load_yolo_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        suffix = Path(model_path).suffix.lower()
        if suffix == '.pt':
            return YOLO(model_path)
        if suffix in ['.onnx', '.engine']:
            return YOLO(model_path, task='detect')
        raise ValueError(f"지원하지 않는 모델 형식입니다: {suffix}")

    def process_frame(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"compressed image 변환 실패: {exc}")
            return

        results = self.model(img, verbose=False)
        plate_count = 0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                cls = int(box.cls[0])
                if not self.show_all_classes and cls != self.plate_class_id:
                    continue

                label = self._get_class_label(cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == self.plate_class_id:
                    plate_count += 1
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{label}: {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        if plate_count > 0:
            status_text = f"Plate detected: {plate_count}"
            status_color = (0, 255, 0)
        else:
            status_text = 'Plate: not detected'
            status_color = (0, 0, 255)

        cv2.putText(
            img,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def _get_class_label(self, cls):
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, f'class_{cls}')
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return f'class_{cls}'

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = YoloTestNode()
    except Exception as exc:
        print(exc)
        if rclpy.ok():
            rclpy.shutdown()
        return

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
