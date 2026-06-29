import json
import time
import math
import threading
import base64
import re

from std_msgs.msg import String

import cv2
import requests

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Bool

from google.cloud import vision
from ultralytics import YOLO


# =========================================================
# 설정값
# =========================================================
SAME_PLATE_WINDOW_SEC = 3.0                       # 같은 번호판으로 볼 시간 윈도우(초)

# 같은 차량(번호판) 추적 중 OCR 재호출 방지용 - 트래픽/API 호출 비용 절약
SAME_CAR_DISTANCE_PX = 50.0
SAME_CAR_RECALL_SEC = 1.0   # 테스트용으로 잠깐 줄이기

GCP_API_KEY = "AIzaSyCFYRBLIvtlXv1J-ngSKl9uwQwjC1pRqgU"   # api key

# YOLO 번호판 검출 모델
YOLO_MODEL_PATH = './semi_allimages_v5n.pt'
PLATE_CLASS_NAME = 'Plate'   # 모델 클래스명 (번호판)
MIN_PLATE_SIZE = 2           # 후에 crop할 영역 최소 가로/세로(px)

# =========================================================
# 토픽명
# =========================================================
AMR1TOOCR = '/robot2/target_plate_image'    # OCR용으로 크롭할 원본 이미지 sub (amr1 -> ocr)
AMR1TOOCR_ID = '/robot2/plate_id'           # event_id sub (amr1 -> ocr)

AMR2TOOCR = '/robot4/target_plate_image'    # OCR용으로 크롭할 원본 이미지 sub (amr2 -> ocr)
AMR2TOOCR_ID = '/robot4/plate_id'           # event_id sub (amr2 -> ocr)
 
MATCH_RESULT = 'match_result'                 # 번호 매치 여부 pub (Boolean) (ocr -> amr2)
MATCH_RESULT_ID = 'match_result_id'           # 시스템 모니터에 띄울 id, 이미지 pub (ocr -> DB, amr2가 발행한거 그대로 string으로 변환)

REGIST = 'regist_car'                       # 차량 번호판 등록 pub (ocr -> server)

REQUEST = 'request_car'                     # id로 차량 번호 조회 요청 pub (ocr -> server)
SERVERTOOCR = 'servertoocr'                 # 등록된 차량 번호 sub (server -> ocr)

DISABLED = 'disabled'                           # 장애인 주차구역에 주차된 차량 정보 (id, 차량 번호) pub (ocr -> server)

DISABLED_RESULT = 'disabled_result'             # 장애인 차량인지 아닌지 (T/F) T: 정상 F: 불법 sub (server -> ocr)
DISABLED_RESULT_AMR = 'disabled_result_amr'    # 장애인 주차구역에서 불법주차인지 정상주차인지 결과 pub (ocr -> amr)
DISABLED_RESULT_ID = 'disabled_result_id'       # 장애인 주차구역에 주차된 차량 정보 (id, 이미지) 불법일때만 string으로 보냄 pub (ocr -> server)

FIRECAR_RESULT = 'firecar_result'               # 소방차인지 아닌지 판단 결과 server로 pub T:정상 F:불법 (ocr -> server)
FIRECAR_RESULT_AMR = 'firecar_result_amr'       # 소방차인지 아닌지 판단 결과 amr로 pub T: 불법 F: 정상
FIRECAR_RESULT_ID = 'firecar_result_id'         # 소방차가 아닌데 소방차전용구역에 주차한 차량 이미지 (불법일때만 보냄) + id도 같이 pub (ocr -> server)

# =========================================================

class PlateOCRNode(Node):

    """
    1. 최초 등록
    - amr1 -> ocr : 원본 이미지, id (TOPIC)
    - 이미지에서 plate 인식하고 plate_box로 crop -> ocr -> 번호판 텍스트 추출
    - ocr -> server : id, 차량 번호 등록 (POST)
    2. 교차 검증
    - amr2 -> ocr : 원본 이미지, id (TOPIC)
    - 이미지에서 plate 인식하고 plate_box로 crop -> ocr -> 번호판 텍스트 추출
    - server -> ocr : id, 차량 번호 (GET)
    - 내가 추출한 것과 서버에 등록된 것과 비교
    - ocr -> server : Bool, id (POST)
    - ocr -> amr2 : Bool (TOPIC)
    """

    def __init__(self):
        super().__init__('plate_ocr_node')
        self.bridge = CvBridge()

        # Google Cloud Vision 클라이언트 (1회만 생성해서 재사용)
        # 환경변수 GOOGLE_APPLICATION_CREDENTIALS에 서비스 계정 json 경로가 설정되어 있어야 함
        # self.vision_client = vision.ImageAnnotatorClient()

        self.vision_client = vision.ImageAnnotatorClient(
                client_options={"api_key": GCP_API_KEY}
        )
        # 권한 이슈로 key 직접 입력

        # ----------YOLO----------
        self.yolo_model = YOLO(YOLO_MODEL_PATH)


        # 최신 이미지를 보관해뒀다가, detect 결과 들어올 때 그 이미지에서 crop
        self.latest_image_1 = None
        self.latest_image_2 = None
        self.event_id_1 = None
        self.event_id_2 = None


        # AMR2 OCR 결과 임시 저장
        self.detected_number = None
        self.detected_image = None
        self.detected_id = None

        # 서버에서 받은 차량 번호
        self.reference_number = None
        self.reference_id = None

        self.zone_number = None
        

        # ----------Publisher----------
        self.match_result = self.create_publisher(Bool, MATCH_RESULT, 10)
        self.match_result_id = self.create_publisher(String, MATCH_RESULT_ID, 10)
        self.register = self.create_publisher(String, REGIST, 10)
        self.request = self.create_publisher(String, REQUEST, 10)
        self.disabled = self.create_publisher(String, DISABLED, 10)
        self.disabled_id = self.create_publisher(String, DISABLED_RESULT_ID, 10)
        self.disabled_amr = self.create_publisher(Bool, DISABLED_RESULT_AMR, 10)
        self.firecar_result = self.create_publisher(Bool, FIRECAR_RESULT, 10)
        self.firecar_amr = self.create_publisher(Bool, FIRECAR_RESULT_AMR, 10)
        self.firecar_result_id = self.create_publisher(String, FIRECAR_RESULT_ID, 10)

        # ----------Subscriber----------
        self.create_subscription(CompressedImage, AMR1TOOCR, self.on_image_1, 10)
        self.create_subscription(CompressedImage, AMR2TOOCR, self.on_image_2, 10)
        self.create_subscription(String, AMR1TOOCR_ID, self.on_id_1, 10)
        self.create_subscription(String, AMR2TOOCR_ID, self.on_id_2, 10)    # json 파일
        self.create_subscription(String, SERVERTOOCR, self.number_subscribed, 10)   # json 파일
        self.create_subscription(Bool, DISABLED_RESULT, self.disabled_result, 10)



        # 같은 번호판 신뢰도 비교용 버퍼
        # self.plate_buffer_1 = {}
        # self.buffer_lock_1 = threading.Lock()
        # self.plate_buffer_2 = {}
        # self.buffer_lock_2 = threading.Lock()


        # 같은 차량(번호판) 추적 중 OCR 재호출 방지용 캐시
        # self.last_ocr_cache = {}

        self.get_logger().info("차량 번호 검출 및 서버 통신 노드 시작")

    # =======================================
    # 이미지 및 ID 수신
    # =======================================
    def on_image_1(self, msg):
        self.latest_image_1 = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f"이미지 수신 성공!")
        self.try_process_amr1()

    def on_id_1(self, msg):
        self.event_id_1 = msg.data
        self.get_logger().info(f"id 수신 성공! : {self.event_id_1}")
        self.try_process_amr1()

    def on_image_2(self, msg):
        self.latest_image_2 = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f"이미지 수신 성공!")
        self.try_process_amr2()
    
    def on_id_2(self, msg):
        data = json.loads(msg.data)
        self.event_id_2 = data['event_id']
        self.zone_number = data['zone']
        self.get_logger().info(f"id 및 zone number 수신 성공! : {self.zone_number}에 주차된 {self.event_id_2}번 차량")
        self.try_process_amr2()


    # =====================================
    # 1. AMR1에서 이미지와 id를 모두 sub 했을 때
    # =====================================
    def try_process_amr1(self):
        # image와 id 중 하나라도 sub하지 않으면 동작 x
        if self.latest_image_1 is None or self.event_id_1 is None:
            return

        self.get_logger().info(f"{self.event_id_1}번 차량 번호 서버 등록 시작")
        self.register_car_number(self.latest_image_1, self.event_id_1)      # 서버 등록 함수 실행

        # 다음 차량 등록을 위해 초기화
        self.latest_image_1 = None
        self.event_id_1 = None
    
    # ==================================================
    # 1. AMR1에서 sub한 이미지에서 차량 번호 검출 후 서버에 등록
    # ==================================================
    def register_car_number(self, img, event_id):
        platebox1 = self.detect_plate_and_crop_img(img, event_id)   # 이미지 편집 함수 실행
        if platebox1 is None:
            return
        
        # OCR 수행
        plate_number1, confidence1 = self.call_vision_ocr(platebox1)

        if plate_number1 is None:
            self.get_logger().warning(f"{event_id}번 차량 번호 검출 실패")
            return
        
        self.get_logger().info(f"{event_id}번 차량 번호: {plate_number1}")
        
        # 서버에 전송
        msg = String()
        data = {
            "event_id": event_id,
            "plate_number": plate_number1
        }
        msg.data = json.dumps(data)
        self.register.publish(msg)
        self.get_logger().info(f"{event_id}번 차량 번호 서버 등록 성공.")
    
    # =====================================
    # 2. AMR2에서 이미지와 id를 모두 sub 했을 때
    # =====================================
    def try_process_amr2(self):
        if self.latest_image_2 is None or self.event_id_2 is None:
            return
        
        self.get_logger().info(f"{self.event_id_2}번 불법 주차 여부 확인 시작")

        # zone number (골목길 or 장애인주차구역 or 소방차전용구역) 에 따라 다른 함수가 실행되도록 함
        # zone number 1 / 2 -> 골목길
        # zone number 3 -> 장애인 주차 구역
        # zone number 4 -> 소방차 전용 구역
        if self.zone_number == '3':
            self.confirm_disabled(self.latest_image_2, self.event_id_2)
        elif self.zone_number == '4':
            self.firecar(self.latest_image_2, self.event_id_2)
        else:
            self.request_car_number(self.latest_image_2, self.event_id_2)

        # 다음 차량 등록을 위해 초기화                                          
        self.latest_image_2 = None
        self.event_id_2 = None
        self.zone_number = None

    # ========================================
    # 2-1. Zone Number 3인 경우 (장애인 주차 구역)
    # ========================================
    """
    장애인 주차 구역의 경우,
    차량 번호 ocr -> 서버로 차량 번호 publish (string) -> 서버에서 장애인 차량인지 판단 -> 서버에서 결과 sub (Bool)
    """
    def confirm_disabled(self, img, event_id):
        platebox2 = self.detect_plate_and_crop_img(img, event_id)   # 이미지 편집 함수 실행
        if platebox2 is None:
            return

        # OCR 수행
        plate_number2, confidence2 = self.call_vision_ocr(platebox2)
    
        if plate_number2 is None:
            self.get_logger().warning(f"{event_id}번 차량 번호 검출 실패")
            return
        
        self.detected_number = plate_number2
        self.detected_image = img
        self.detected_id = event_id

        msg = String()
        msg.data = self.detected_number
        self.disabled.publish(msg)  # 서버로 차량 번호 publish
        self.get_logger().info(f"{event_id}번 장애인 차량 등록 여부 확인 요청")
    
    
    # ============================================
    # 2-1-1. Server로부터 장애인 차량 등록 여부 수신 완료
    # ============================================
    def disabled_result(self, msg):
        self.get_logger().info(f"서버로부터 장애인 차량 등록 여부 수신 성공!")
        server_result = msg.data        # 서버로부터 구독한 결과 (T: 정상주차, F: 불법주차)
        amr_result = not server_result  # amr로 발행할 결과 (T: 불법주차, F: 정상주차)

        bool_msg = Bool()
        bool_msg.data = amr_result
        self.disabled_amr.publish(bool_msg) # amr로 불법주차인지 (T) 정상주차인지 (F) pub

        # 장애인 주차 구역에서 불법주차인 경우 서버와의 통신
        # server_result == False
        if not server_result:
            success, buffer = cv2.imencode(".jpg", self.detected_image)

            if not success:
                self.get_logger().warning("Image Encoding Failed")
                self.clear_detected_data()
                return
            
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            payload = {
                "event_id": self.detected_id,
                "car_number": self.detected_number,
                "image": image_base64
            }

            result_msg = String()
            result_msg.data = json.dumps(payload)

            self.disabled_id.publish(result_msg)    # 서버로 id와 차량 번호, 차량 이미지 pub (T/F는 서버에서 이미 판단)

            self.get_logger().info("Disabled Result Published")


        self.clear_detected_data()

    # =======================================
    # 2-2. Zone Number 4인 경우 (소방차 전용구역)
    # =======================================
    """
    소방차 전용 구역의 경우,
    차량 번호 ocr -> 소방차 번호인지 판단 -> amr로 결과 pub (T/F) -> 서버로 결과 pub (T/F)
    불법주차인 경우, 서버로 차량 id, 차량 번호, image pub (string)
    """
    def firecar(self, img, event_id):
        platebox2 = self.detect_plate_and_crop_img(img, event_id)
        if platebox2 is None:
            return

        # OCR 수행
        plate_number2, confidence2 = self.call_vision_ocr(platebox2)
    
        if plate_number2 is None:
            self.get_logger().warning(f"{event_id}번의 차량 번호 검출 실패")
            return
        
        self.detected_number = plate_number2
        self.detected_image = img
        self.detected_id = event_id
        
        normalized = self.normalize_plate(self.detected_number)

        # is_firecar = (
        #     normalized.startswith("998") or
        #     normalized.startswith("999") or
        #     normalized.startswith("98")  or
        #     normalized.startswith("99")
        # )
                    
        # server_msg = Bool()
        # server_msg.data = False
        # self.firecar_result.publish(server_msg)
        # self.get_logger().info("Firecar Result Published")

        # amr_msg = Bool()
        # amr_msg.data = not is_firecar
        # self.firecar_amr.publish(amr_msg)

       
        # 소방차 전용 구역에서 불법주차인 경우 서버와의 통신
        # is_firecar == False
        # if not is_firecar:
        success, buffer = cv2.imencode(".jpg", self.detected_image)
        
        if not success:
            self.get_logger().warning("Image Encoding Failed")
            return
        
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "event_id": self.detected_id,
            "car_number": self.detected_number,
            "image": image_base64
        }

        result_msg = String()
        result_msg.data = json.dumps(payload)

        self.firecar_result_id.publish(result_msg)

        self.get_logger().info("Firecar Result ID Published")
        
        self.clear_detected_data()

    # =======================================================================
    # 2-3. (골목길 불법주차) AMR2에서 sub한 이미지에서 차량 번호 검출 후 서버에 id에 해당하는 차량 번호 요청
    # =======================================================================
    def request_car_number(self, img, event_id):
        platebox2 = self.detect_plate_and_crop_img(img, event_id)
        if platebox2 is None:
            return

        # OCR 수행
        plate_number2, confidence2 = self.call_vision_ocr(platebox2)
    
        if plate_number2 is None:
            self.get_logger().warning(f"{event_id}번의 차량 번호 검출 실패")
            return
        
        self.detected_number = plate_number2
        self.detected_image = img
        self.detected_id = event_id
        
        msg = String()
        payload = {
            "event_id": event_id
        }
        msg.data = json.dumps(payload)
        self.request.publish(msg)
        self.get_logger().info(f"{event_id}번의 차량 번호 요청")

    
    # ================================================
    # 2-4. 서버에서 id에 해당하는 차량 번호 subscribe 했을 때
    # ================================================
    def number_subscribed(self, msg):
        data = json.loads(msg.data)
        self.reference_id = data["event_id"]
        self.reference_number = data["plate_number"]
        if self.reference_number is None:
            self.get_logger().info(f"서버로부터 차량 번호 수신 실패")
            return
        self.get_logger().info(f"서버로부터 차량 번호 수신 성공!")
        self.verify_number()

    def verify_number(self):   
        if self.detected_number is None:
            return
        if self.reference_number is None:
            return
        
        if self.detected_id != self.reference_id: 
            self.get_logger().warning( "Event ID mismatch" ) 
            self.clear_detected_data()
            self.reference_number = None
            self.reference_id = None
            return
        
        
        is_match = (self.normalize_plate(self.detected_number) == self.normalize_plate(self.reference_number))
                    
        bool_msg = Bool()
        bool_msg.data = is_match
        self.match_result.publish(bool_msg)
        self.get_logger().info("Match Result Published")

        if is_match:
            success, buffer = cv2.imencode(".jpg", self.detected_image)
            if not success:
                self.get_logger().warning("Image Encoding Failed")
                self.clear_detected_data()
                return
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            payload = {
                "event_id": self.detected_id,
                "matched": is_match,
                "image": image_base64
            }
            image_msg = String()
            image_msg.data = json.dumps(payload)
            self.match_result_id.publish(image_msg)
            self.get_logger().info("Match Image Published")

        self.clear_detected_data()
        self.reference_number = None
        self.reference_id = None


    # =======================================
    # 번호판 인식과 OCR을 위한 크롭 이미지 생성
    # =======================================
    def detect_plate_and_crop_img(self, img, event_id):
        results = self.yolo_model(img, verbose=False)
        plate_box = None

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]
                if label != PLATE_CLASS_NAME:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (x2 - x1) < MIN_PLATE_SIZE:
                    continue

                if (y2 - y1) < MIN_PLATE_SIZE:
                    continue

                plate_box = (x1, y1, x2, y2)

                break

            # 번호판을 찾으면 바깥 반복문도 종료
            if plate_box is not None:
                break

            # 끝까지 탐색했는데 번호판이 없으면
        if plate_box is None:
            self.get_logger().warning(f"{event_id}번 차량 번호판 검출 실패")
            return None

        x1, y1, x2, y2 = plate_box

        margin = 30

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.shape[1], x2 + margin)
        y2 = min(img.shape[0], y2 + margin)

        crop = img[y1:y2, x1:x2]

        crop = cv2.resize(
            crop,
            None,
            fx=8,
            fy=8,
            interpolation=cv2.INTER_CUBIC
        )


        self.get_logger().info(f"[{event_id}] 번호판 검출 성공")    
        return crop
    
    # =======================================
    # Google Vision OCR
    # =======================================
    def call_vision_ocr(self, crop_img):

        success, encoded = cv2.imencode(".jpg", crop_img)

        if not success:

            self.get_logger().warning("OCR_FAIL : Image Encode")

            return None, None

        image = vision.Image(content=encoded.tobytes())

        image_context = vision.ImageContext(language_hints=["ko"])

        try:
            response = self.vision_client.document_text_detection(
                image=image,
                image_context=image_context
            )

        except Exception as e:

            self.get_logger().warning(f"OCR_FAIL : {e}")

            return None, None

        if response.error.message:

            self.get_logger().warning(response.error.message)

            return None, None

        if not response.text_annotations:

            self.get_logger().warning("OCR_FAIL : No Text")

            return None, None

        plate_text = response.text_annotations[0].description

        # self.get_logger().info(f"RAW OCR : {repr(plate_text)}")

        plate_text = self.normalize_plate(plate_text)
        self.get_logger().info(f"Normalized : {plate_text}")

        confidence = []

        if response.full_text_annotation.pages:

            for page in response.full_text_annotation.pages:

                for block in page.blocks:

                    for paragraph in block.paragraphs:

                        for word in paragraph.words:

                            for symbol in word.symbols:

                                if symbol.confidence is not None:

                                    confidence.append(symbol.confidence)

        if confidence:

            ocr_conf = sum(confidence) / len(confidence)

        else:

            ocr_conf = None

        return plate_text, ocr_conf
    

    # =========================================================
    # 번호판 문자열 정규화
    #
    # 비교를 위해 공백 제거 후
    # 출력은 12가 3456 / 123가 4567 형태로 맞춘다.
    # =========================================================
    def normalize_plate(self, text):

        if text is None:
            return None

        # 줄바꿈 제거
        text = text.replace("\n", "")

        # 공백 제거
        text = text.replace(" ", "")

        # 특수문자 제거
        text = re.sub(r"[^0-9가-힣]", "", text)

        # 12가3456
        m = re.fullmatch(r"(\d{2})([가-힣])(\d{4})", text)
        if m:
            return f"{m.group(1)}{m.group(2)} {m.group(3)}"

        # 123가4567
        m = re.fullmatch(r"(\d{3})([가-힣])(\d{4})", text)
        if m:
            return f"{m.group(1)}{m.group(2)} {m.group(3)}"

        return text
        
    # =========================================================
    # 초기화 함수
    # =========================================================
    def clear_detected_data(self):
        self.detected_number = None
        self.detected_image = None
        self.detected_id = None

def main(args=None):
    rclpy.init(args=args)
    node = PlateOCRNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()