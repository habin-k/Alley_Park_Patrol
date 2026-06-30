import base64
import requests
import cv2
import numpy as np

SERVER = 'http://192.168.107.42:8000'
DB_ID  = 1  # 확인할 parking_events.id


def main():
    resp = requests.get(f'{SERVER}/api/monitor/events/', timeout=5)
    events = resp.json()

    target = next((e for e in events if e['event_id'] == DB_ID), None)
    if target is None:
        print(f"event_id={DB_ID} 없음")
        return

    vi = target.get('vehicle_info')
    if not vi or not vi.get('ocr_image'):
        print("ocr_image 없음")
        return

    img_bytes = base64.b64decode(vi['ocr_image'])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    cv2.imshow(f"ocr_image (event_id={DB_ID})", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
