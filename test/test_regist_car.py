"""
test_regist_car.py — AMR1 번호판 등록 테스트 (ROS2 없이 단독 실행)

plate_ocr_node.py : register_car_number() → regist_car 토픽 발행
bridge_ocr.py     : on_regist_car() → POST /api/vehicle/

가정:
  - YOLO로 번호판 크롭 완료 (크롭된 이미지 경로 입력)
  - parking_events에 해당 db_id 레코드 존재 (status=DETECTED)

테스트 흐름:
  OCR → 번호판 텍스트 추출
      → POST /api/vehicle/ (event_id + plate_number)
      → vehicle_info 저장 + status=SCANNED

사용법:
  python3 test_regist_car.py <이미지경로> <db_id>

  예시:
    python3 test_regist_car.py /home/rokey/Downloads/plate.jpg 1
"""

import sys
import re
import requests
import cv2
from google.cloud import vision

SERVER      = 'http://192.168.107.42:8000'
GCP_API_KEY = "AIzaSyCFYRBLIvtlXv1J-ngSKl9uwQwjC1pRqgU"


# ── 번호판 텍스트 정규화 ──────────────────────────────────────
def normalize_plate(text):
    if text is None:
        return None
    text = text.replace("\n", "").replace(" ", "")
    text = re.sub(r"[^0-9가-힣]", "", text)

    m = re.fullmatch(r"(\d{2})([가-힣])(\d{4})", text)
    if m:
        return f"{m.group(1)}{m.group(2)} {m.group(3)}"

    m = re.fullmatch(r"(\d{3})([가-힣])(\d{4})", text)
    if m:
        return f"{m.group(1)}{m.group(2)} {m.group(3)}"

    return text


# ── Google Vision OCR ─────────────────────────────────────────
def run_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지를 읽을 수 없음: {image_path}")
        return None

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        print("[ERROR] 이미지 인코딩 실패")
        return None

    client   = vision.ImageAnnotatorClient(client_options={"api_key": GCP_API_KEY})
    image    = vision.Image(content=encoded.tobytes())
    response = client.text_detection(image=image)

    if response.error.message:
        print(f"[ERROR] OCR 오류: {response.error.message}")
        return None

    if not response.text_annotations:
        print("[ERROR] 번호판 텍스트 검출 실패")
        return None

    raw_text = response.text_annotations[0].description
    print(f"[OCR] RAW:   {repr(raw_text)}")

    plate_text = normalize_plate(raw_text)
    print(f"[OCR] 정규화: {plate_text}")

    return plate_text


# ── 서버에 번호판 등록 ────────────────────────────────────────
def regist_car(db_id, plate_number):
    resp = requests.post(f"{SERVER}/api/vehicle/", json={
        "event_id":     db_id,
        "plate_number": plate_number,
    }, timeout=5)
    if resp.status_code == 201:
        print(f"[SERVER] 번호판 등록 성공: {resp.json()}")
    else:
        print(f"[ERROR] 등록 실패: {resp.status_code} {resp.text}")


# ── 메인 ──────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("사용법: python3 test_regist_car.py <이미지경로> <db_id>")
        print("예시:   python3 test_regist_car.py /home/rokey/Downloads/plate.jpg 1")
        sys.exit(1)

    image_path = sys.argv[1]
    db_id      = int(sys.argv[2])

    print(f"\n=== AMR1 번호판 등록 테스트 ===")
    print(f"이미지: {image_path}")
    print(f"db_id : {db_id}\n")

    # 1. OCR
    plate_text = run_ocr(image_path)
    if plate_text is None:
        print("[FAIL] OCR 실패 — 종료")
        sys.exit(1)

    # 2. 서버에 번호판 등록
    print(f"\n[ACTION] 번호판 등록 전송: event_id={db_id}, plate={plate_text}")
    regist_car(db_id, plate_text)


if __name__ == "__main__":
    main()
