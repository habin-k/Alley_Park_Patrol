import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class CameraSource:									                # 카메라의 설정과 객체를 담는 보조 클래스를 정의합니다.
    def __init__(
            self, name, index, calibration_path, homography, capture,
            annotated_image_topic):
        self.name = name
        self.index = index
        self.calibration_path = calibration_path					
        self.homography = homography							    # 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
        self.capture = capture
        self.annotated_image_topic = annotated_image_topic
        self.annotated_image_publisher = None


class MultiWebcamObjectMapper(Node):							# 여러 웹캠 감지 결과를 맵 좌표로 publish하는 ROS2 노드를 정의합니다.
    """Detect objects from multiple fixed webcams and publish map coordinates."""

    WINDOW_PREFIX = 'multi_webcam_object_mapper2'

    def __init__(self):
        super().__init__('multi_webcam_object_mapper2')					    # ROS 노드 이름
        self._declare_parameters()							                # 노드에서 사용할 ROS 파라미터선언

        self.model = self._load_yolo_model()						        # YOLO 객체 감지 모델을 불러옵니다.
        self.class_names = self._model_class_names()					    # 모델의 클래스 ID와 이름 목록을 가져옵니다.
        self.confidence_threshold = float(						            # 객체 감지 결과를 사용할 최소 신뢰도를 다룹니다.
            self.get_parameter('confidence_threshold').value)
        self.class_ids = self._parse_int_list(						            # 감지할 클래스 ID 필터 목록을 다룹니다.
            self.get_parameter('class_ids').value)
        self.show_image = bool(self.get_parameter('show_image').value)			# 카메라 영상 창 표시 여부를 다룹니다.
        self.publish_topic = str(self.get_parameter('publish_topic').value)		# 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.fusion_distance = max(							                    # 여러 감지를 같은 객체로 합칠 거리 기준을 다룹니다.
            0.0, float(self.get_parameter('fusion_distance').value))

        self.cameras = self._open_cameras()						                # 설정된 모든 카메라를 열고 준비합니다.
        for camera in self.cameras:
            camera.annotated_image_publisher = self.create_publisher(
                Image, camera.annotated_image_topic, 10)
        self.publisher = self.create_publisher(						            
            String, self.publish_topic, 10)						                # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.processing = False								                    # 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.
        self.create_timer(
            float(self.get_parameter('frame_period').value),
            self.process_frames,
        )
        self.get_logger().info(
            f'Ready with {len(self.cameras)} webcams; '
            f'publishing {self.publish_topic}')			                        # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.

    def _declare_parameters(self):							# ROS 파라미터 기본값들을 선언하는 함수를 정의합니다.
        self.declare_parameter('camera_indices', [0, 2])				        # 사용할 카메라 장치 번호 기본값을 선언합니다.
        self.declare_parameter('camera_names', ['webcam1', 'webcam2'])			# 카메라 이름 기본값을 선언합니다.
        self.declare_parameter('annotated_image_topics', [
            '/webcam_images/webcam1/detections',
            '/webcam_images/webcam2/detections',
        ])
        self.declare_parameter('calibration_paths', ['', ''])				    # 카메라별 캘리브레이션 파일 경로 기본값을 선언합니다.
        self.declare_parameter('model_path', '')					            # YOLO 모델 파일 경로 파라미터를 선언합니다.
        self.declare_parameter('class_ids', [])						            # 필터링할 클래스 ID 목록 파라미터를 선언합니다.
        self.declare_parameter('confidence_threshold', 0.70)				    # 객체 감지 최소 신뢰도 기본값을 선언합니다.
        self.declare_parameter('frame_period', 0.1)					            # 프레임 처리 주기 기본값을 초 단위로 선언합니다.
        self.declare_parameter('show_image', True)					            # 영상 출력 여부 기본값을 선언합니다.
        self.declare_parameter(
            'publish_topic', '/webcam_objects/map_detections')				    # 감지 결과 JSON을 보낼 ROS 토픽 이름을 다룹니다.
        self.declare_parameter('fusion_distance', 0.25)					        # 감지 결과를 합칠 거리 기준 기본값을 선언합니다.

    def _load_yolo_model(self):							        # YOLO 모델을 불러오는 함수를 정의합니다.
        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        if not model_path:
            raise RuntimeError('model_path is required')
        if not os.path.isfile(model_path):
            raise RuntimeError(f'YOLO model not found: {model_path}')
        from ultralytics import YOLO
        return YOLO(model_path)

    def _model_class_names(self):							# 모델 클래스 이름 목록을 정리하는 함수를 정의합니다.
        names = getattr(self.model, 'names', {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def _open_cameras(self):					            # 설정된 카메라들을 여는 함수를 정의합니다.
        indices = self._parse_int_list(
            self.get_parameter('camera_indices').value)					            # 사용할 카메라 장치 번호 목록을 다룹니다.
        names = self._parse_str_list(self.get_parameter('camera_names').value)		# 카메라 이름 목록을 다룹니다.
        annotated_image_topics = self._parse_str_list(
            self.get_parameter('annotated_image_topics').value)
        calibration_paths = self._parse_str_list(					                # 카메라 캘리브레이션 파일 경로 목록을 다룹니다.
            self.get_parameter('calibration_paths').value)

        if len(indices) != len(calibration_paths):					                # 카메라 캘리브레이션 파일 경로 목록을 다룹니다.
            raise RuntimeError(
                'camera_indices and calibration_paths must have same length')
        if names and len(names) != len(indices):
            raise RuntimeError(
                'camera_names must be empty or match camera_indices length')
        if (annotated_image_topics
                and len(annotated_image_topics) != len(indices)):
            raise RuntimeError(
                'annotated_image_topics must be empty or match '
                'camera_indices length')

        cameras = []									                        # 열린 카메라 정보를 담을 리스트를 만듭니다.
        for camera_number, index in enumerate(indices):
            name = names[camera_number] if names else f'webcam{camera_number + 1}'
            annotated_image_topic = (
                annotated_image_topics[camera_number]
                if annotated_image_topics
                else f'/webcam_images/{name}/detections')
            calibration_path = calibration_paths[camera_number]
            homography = self._load_homography(calibration_path, name)			# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
            capture = cv2.VideoCapture(index)
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open {name} at camera index {index}')
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(
                f'{name}: index={index}, size={width}x{height}, '
                f'annotated_image_topic={annotated_image_topic}')
            cameras.append(CameraSource(
                name, index, calibration_path, homography, capture,
                annotated_image_topic))
        return cameras

    def _bgr_to_image_message(self, frame, frame_id):                           # OpenCV BGR 이미지를 ROS2 Image 메시지로 변환 (OpenCV이미지는 Topic으로 바로 보낼 수 없기 때문에, sensor_msgs/image형식으로 변환
        height, width, channels = frame.shape
        if channels != 3:
            raise RuntimeError('expected BGR image with 3 channels')

        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.height = height
        message.width = width
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = width * channels
        message.data = frame.tobytes()
        return message

    def _load_homography(self, path, camera_name):					            # 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
        expanded_path = os.path.expanduser(str(path))					        
        if not expanded_path:
            raise RuntimeError(f'calibration_path is required for {camera_name}')
        with open(expanded_path, encoding='utf-8') as stream:				    # 컨텍스트 관리 블록을 시작합니다.
            calibration = json.load(stream)						                # JSON 파일 내용을 파이썬 dict로 읽습니다.

        image_points = np.asarray(							                # 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
            calibration.get('image_points', []), dtype=np.float32)
        map_points = np.asarray(							                # 캘리브레이션에 사용할 맵 기준점들을 다룹니다.
            calibration.get('map_points', []), dtype=np.float32)
        if image_points.shape != map_points.shape:
            raise RuntimeError(
                f'{camera_name}: image_points and map_points shape mismatch')
        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise RuntimeError(
                f'{camera_name}: calibration points must be [x, y] pairs')
        if len(image_points) < 4:
            raise RuntimeError(
                f'{camera_name}: at least four calibration pairs are required')

        homography, mask = cv2.findHomography(						# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
            image_points, map_points, cv2.RANSAC)					# 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
        if homography is None:								# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
            raise RuntimeError(f'{camera_name}: failed to calculate homography')	
        inliers = int(mask.sum()) if mask is not None else len(image_points)		# 캘리브레이션에 사용할 이미지 기준점들을 다룹니다.
        self.get_logger().info(
            f'{camera_name}: loaded {len(image_points)} calibration pairs '
            f'({inliers} inliers)')
        return homography								# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 반환

    def process_frames(self):								# 타이머마다 카메라 프레임을 처리하는 (핵심)함수
        if self.processing:
            return
        self.processing = True
        try:
            detections = []								                        # 감지 결과를 담을 리스트를 만듭니다.
            for camera in self.cameras:
                success, frame = camera.capture.read()		                    # 카메라에서 현재 프레임을 읽습니다.
                if not success:
                    self.get_logger().warn(					                    
                        f'{camera.name}: failed to read frame',
                        throttle_duration_sec=2.0)					            
                    continue								                    # 현재 반복을 건너뛰고 다음 반복으로 넘어갑니다.
                camera_detections = self._detect_objects(camera, frame)			
                detections.extend(camera_detections)					        
                self._draw_detections(frame, camera_detections)
                camera.annotated_image_publisher.publish(
                    self._bgr_to_image_message(frame, camera.name))
                if self.show_image:							                    # 카메라 영상 창 표시 여부를 다룹니다.
                    cv2.imshow(f'{self.WINDOW_PREFIX} {camera.name}', frame)	# OpenCV 창에 현재 영상을 표시합니다.

            fused_objects = self._fuse_detections(detections)
            payload = {
                'stamp': self.get_clock().now().nanoseconds / 1e9,
                'system_time': time.time(),
                'frame_id': 'map',
                'detections': detections,
                'objects': fused_objects,	
            }
            self._publish_payload(payload)
            if self.show_image:
                cv2.waitKey(1)								# OpenCV 창 갱신과 키 입력 처리를 수행합니다.
        except Exception as error:
            self.get_logger().error(f'Frame processing failed: {error}')
        finally:									        # 성공 여부와 관계없이 항상 실행할 정리 블록입니다.
            self.processing = False							# 프레임 처리 중복 실행을 막기 위한 상태값을 초기화합니다.

    def _detect_objects(self, camera, frame):						# 한 카메라 프레임에서 객체를 감지하는 함수를 정의합니다.
        detections = []
        for result in self.model(frame, verbose=False):
            for box in result.boxes:
                class_id = int(box.cls[0])
                if self.class_ids and class_id not in self.class_ids:
                    continue
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:				# 객체 감지 결과를 사용할 최소 신뢰도를 다룹니다.
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                pixel_x = (x1 + x2) / 2.0
                pixel_y = y2
                map_x, map_y = self._pixel_to_map(
                    camera.homography, pixel_x, pixel_y)				# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
                detections.append({
                    'camera': camera.name,
                    'camera_index': camera.index,
                    'class_id': class_id,
                    'class_name': self.class_names.get(
                        class_id, f'class_{class_id}'),
                    'confidence': confidence,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'pixel': {'x': pixel_x, 'y': pixel_y},
                    'map': {'x': map_x, 'y': map_y},
                })
        return detections

    def _pixel_to_map(self, homography, pixel_x, pixel_y):				# 픽셀 좌표를 맵 좌표로 바꾸는 호모그래피 행렬을 다룹니다.
        pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        map_point = cv2.perspectiveTransform(pixel, homography)
        return float(map_point[0, 0, 0]), float(map_point[0, 0, 1])

    def _fuse_detections(self, detections):						# 여러 감지 결과를 같은 객체 단위로 합치는 함수를 정의합니다.
        objects = []
        for detection in sorted(
                detections, key=lambda item: item['confidence'], reverse=True):
            match = self._find_matching_object(objects, detection)
            if match is None:
                objects.append({
                    'object_id': len(objects),
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'map': dict(detection['map']),
                    'sources': [self._source_summary(detection)],
                })
                continue

            sources = match['sources']
            sources.append(self._source_summary(detection))
            total_confidence = sum(source['confidence'] for source in sources)
            match['map']['x'] = sum(
                source['map']['x'] * source['confidence']
                for source in sources) / total_confidence
            match['map']['y'] = sum(
                source['map']['y'] * source['confidence']
                for source in sources) / total_confidence
            match['confidence'] = max(
                match['confidence'], detection['confidence'])
        return objects

    def _find_matching_object(self, objects, detection):			# 현재 감지와 같은 객체로 볼 기존 객체를 찾는 함수를 정의합니다.
        if self.fusion_distance <= 0.0:							    # 여러 감지를 같은 객체로 합칠 거리 기준을 다룹니다.
            return None
        for object_data in objects:
            if object_data['class_id'] != detection['class_id']:
                continue
            dx = object_data['map']['x'] - detection['map']['x']
            dy = object_data['map']['y'] - detection['map']['y']
            if (dx * dx + dy * dy) ** 0.5 <= self.fusion_distance:
                return object_data
        return None

    def _source_summary(self, detection):						    # 원본 감지 정보를 간단히 요약하는 함수를 정의합니다.
        return {
            'camera': detection['camera'],
            'camera_index': detection['camera_index'],
            'confidence': detection['confidence'],
            'pixel': dict(detection['pixel']),
            'map': dict(detection['map']),
        }

    def _draw_detections(self, frame, detections):					# 감지 결과를 영상 프레임 위에 그리는 함수를 정의합니다.
        for detection in detections:
            x1, y1, x2, y2 = [
                int(value) for value in detection['bbox_xyxy']]
            map_x = detection['map']['x']
            map_y = detection['map']['y']
            label = (
                f"{detection['class_name']} {detection['confidence']:.2f} "
                f"map=({map_x:.2f},{map_y:.2f})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(
                frame,
                (int(detection['pixel']['x']), int(detection['pixel']['y'])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                label,	
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def _publish_payload(self, payload):						# 감지 결과 payload를 publish하는 함수를 정의합니다.
        message = String()
        message.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(message)

    def _parse_int_list(self, value):						# 파라미터 값을 정수 리스트로 변환하는 함수를 정의합니다.
        if isinstance(value, str):							
            if not value.strip():							
                return []								
            return [int(item.strip()) for item in value.split(',')]			
        return [int(item) for item in value]						

    def _parse_str_list(self, value):							# 파라미터 값을 문자열 리스트로 변환하는 함수를 정의합니다.
        if isinstance(value, str):							
            if not value.strip():							
                return []								
            return [item.strip() for item in value.split(',')]				
        return [str(item) for item in value]						

    def destroy_node(self):								# 노드 종료 시 리소스를 정리하는 함수를 정의합니다.
        for camera in getattr(self, 'cameras', []):					
            camera.capture.release()							
        cv2.destroyAllWindows()								
        super().destroy_node()								


def main(args=None):
    rclpy.init(args=args)								# ROS2 파이썬 클라이언트를 초기화합니다.
    node = None										    # 노드 변수를 먼저 None으로 초기화합니다.
    try:										
        node = MultiWebcamObjectMapper()						
        rclpy.spin(node)								# ROS2 노드의 콜백이 계속 실행되게 합니다.
    except KeyboardInterrupt:							# Ctrl+C로 종료하는 경우를 처리합니다.
        pass
    except Exception as error:
        if node is not None:								
            node.get_logger().fatal(str(error))						
        else:										
            print(f'Failed to start multi_webcam_object_mapper2: {error}')		
    finally:
        if node is not None:								
            node.destroy_node()								
        if rclpy.ok():									
            rclpy.shutdown()								# ROS2 클라이언트를 종료합니다.


if __name__ == '__main__':
    main()
