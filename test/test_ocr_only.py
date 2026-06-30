import sys
import re
import cv2
from google.cloud import vision

GCP_API_KEY = "AIzaSyCFYRBLIvtlXv1J-ngSKl9uwQwjC1pRqgU"


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


def run_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지를 읽을 수 없음: {image_path}")
        return

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        print("[ERROR] 이미지 인코딩 실패")
        return

    client   = vision.ImageAnnotatorClient(client_options={"api_key": GCP_API_KEY})
    image    = vision.Image(content=encoded.tobytes())
    response = client.text_detection(image=image)

    if response.error.message:
        print(f"[ERROR] OCR 오류: {response.error.message}")
        return

    if not response.text_annotations:
        print("[ERROR] 텍스트 검출 실패")
        return

    raw_text = response.text_annotations[0].description
    print(f"[RAW]        {repr(raw_text)}")

    normalized = normalize_plate(raw_text)
    print(f"[정규화]     {normalized}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python3 test_ocr_only.py <이미지경로>")
        print("예시:   python3 test_ocr_only.py /home/rokey/Downloads/plate.jpg")
        sys.exit(1)
    run_ocr(sys.argv[1])
