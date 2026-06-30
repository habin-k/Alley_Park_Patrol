"""
test_plate_match.py — AMR2 OCR plate-match 테스트 (ROS2 없이 단독 실행)

가정:
  - AMR1이 이미 번호판 등록 완료 (vehicle_info에 plate_number 저장, status=SCANNED)
  - AMR2가 이미 YOLO로 번호판 크롭 완료

테스트 범위:
  Google Vision OCR → 텍스트 추출 → 서버 번호판 조회 → 비교 → 결과 전송

사용법:
  python3 test_plate_match.py <이미지경로> <db_id>

  예시:
    python3 test_plate_match.py /home/rokey/plate.jpg 3
"""

import sys
import re
import base64
import requests
import cv2
from google.cloud import vision

SERVER    = 'http://192.168.107.42:8000'
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
        return None, None

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        print("[ERROR] 이미지 인코딩 실패")
        return None, None

    client   = vision.ImageAnnotatorClient(client_options={"api_key": GCP_API_KEY})
    image    = vision.Image(content=encoded.tobytes())
    response = client.text_detection(image=image)

    if response.error.message:
        print(f"[ERROR] OCR 오류: {response.error.message}")
        return None, None

    if not response.text_annotations:
        print("[ERROR] 번호판 텍스트 검출 실패")
        return None, None

    raw_text = response.text_annotations[0].description
    print(f"[OCR] RAW: {repr(raw_text)}")

    plate_text = normalize_plate(raw_text)
    print(f"[OCR] 정규화: {plate_text}")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return plate_text, image_b64


# ── 서버에서 AMR1 등록 번호판 조회 ───────────────────────────
def get_server_plate(db_id):
    resp = requests.get(f"{SERVER}/api/vehicle/{db_id}/", timeout=5)
    if resp.status_code == 200:
        plate = resp.json().get("plate_number")
        print(f"[SERVER] 저장된 번호판: {plate}")
        return plate
    print(f"[ERROR] 번호판 조회 실패: {resp.status_code} {resp.text}")
    return None


# ── 결과 전송 ─────────────────────────────────────────────────
def send_match(db_id, image_b64):
    resp = requests.post(f"{SERVER}/api/vehicle/verify/", json={
        "event_id":  db_id,
        "match":     True,
        "ocr_image": image_b64,
    }, timeout=5)
    if resp.status_code == 200:
        print(f"[SERVER] WARNING_ISSUED 처리 성공: {resp.json()}")
    else:
        print(f"[ERROR] verify 실패: {resp.status_code} {resp.text}")


def send_no_match(db_id):
    resp = requests.post(f"{SERVER}/api/vehicle/verify/", json={
        "event_id": db_id,
        "match":    False,
    }, timeout=5)
    if resp.status_code == 200:
        print(f"[SERVER] 이벤트 삭제 성공: {resp.json()}")
    else:
        print(f"[ERROR] 삭제 실패: {resp.status_code} {resp.text}")


# ── 메인 ──────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("사용법: python3 test_plate_match.py <이미지경로> <db_id>")
        print("예시:   python3 test_plate_match.py /home/rokey/Downloads/plate.jpg 1")
        sys.exit(1)

    image_path = sys.argv[1]
    db_id      = int(sys.argv[2])

    print(f"\n=== AMR2 plate-match 테스트 ===")
    print(f"이미지: {image_path}")
    print(f"db_id : {db_id}\n")

    # 1. OCR
    ocr_text, image_b64 = run_ocr(image_path)
    if ocr_text is None:
        print("[FAIL] OCR 실패 — 종료")
        sys.exit(1)

    # 2. 서버 번호판 조회
    server_text = get_server_plate(db_id)
    if server_text is None:
        print("[FAIL] 서버 번호판 조회 실패 — 종료")
        sys.exit(1)

    server_text_norm = normalize_plate(server_text)

    # 3. 비교
    is_match = (ocr_text == server_text_norm)
    print(f"\n[MATCH] OCR={ocr_text}  |  SERVER={server_text_norm}  |  결과={'일치 ✓' if is_match else '불일치 ✗'}")

    # 4. 결과 전송
    if is_match:
        print("[ACTION] WARNING_ISSUED 전송")
        send_match(db_id, image_b64)
    else:
        print("[ACTION] 이벤트 삭제 전송")
        send_no_match(db_id)


if __name__ == "__main__":
    main()
