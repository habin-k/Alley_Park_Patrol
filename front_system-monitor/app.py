from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import requests
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 세션 암호화용 - 운영시 환경변수로 교체 권장

# ----------------------------------------
# 하드코딩 로그인 정보 (요구사항대로)
# ----------------------------------------
USERNAME = 'user'
PASSWORD = 'password'

# ----------------------------------------
# 백엔드 API 주소 (Swagger 기준)
# ----------------------------------------
API_BASE_URL = 'http://192.168.107.42:8000'
EVENTS_ENDPOINT = f'{API_BASE_URL}/api/monitor/events/'
WEBCAM1_ENDPOINT = f'{API_BASE_URL}/api/webcam1/frame/latest/'
DISABLED_ENDPOINT = f'{API_BASE_URL}/api/monitor/disabled/'


def get_violations(status_filter=None):
    """백엔드 /api/monitor/events/ 에서 위반(주차 단속) 이벤트 리스트 조회"""
    params = {}
    if status_filter:
        params['status'] = status_filter

    try:
        response = requests.get(EVENTS_ENDPOINT, params=params, timeout=5)
        response.raise_for_status()
        print("받는 데이터 내용 확인:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        # 백엔드 서버가 꺼져있거나 응답이 없을 때 빈 리스트로 처리
        print(f"[ERROR] 이벤트 API 호출 실패: {e}")
        return []


def get_disabled_list():
    """백엔드 /api/monitor/disabled/ 에서 장애인 차량 리스트 조회"""
    try:
        response = requests.get(DISABLED_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 장애인 차량 리스트 조회 실패: {e}")
        return []


def register_disabled(plate_number):
    """백엔드에 장애인 차량 번호판 등록"""
    try:
        response = requests.post(
            f'{DISABLED_ENDPOINT}register/',
            json={'plate_number': plate_number},
            timeout=5
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 장애인 차량 등록 실패: {e}")
        return False


def delete_disabled(plate_number):
    """백엔드에서 장애인 차량 번호판 삭제"""
    try:
        response = requests.delete(f'{DISABLED_ENDPOINT}{plate_number}/', timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 장애인 차량 삭제 실패: {e}")
        return False


# ----------------------------------------
# AMR1 카메라 - ROS2 Subscriber (compressed 토픽 구독)
# Flask 프로세스 안에서 별도 스레드로 rclpy spin을 돌리고,
# 콜백에서 최신 프레임(JPEG bytes)을 amr1_latest_frame에 저장한다.
# ----------------------------------------
AMR1_IMAGE_TOPIC = '/robot2/oakd/rgb/image_raw/compressed'  # 실제 확인된 토픽명

amr1_latest_frame = None
amr1_frame_lock = threading.Lock()


class Amr1CameraSubscriber(Node):
    def __init__(self):
        super().__init__('amr1_camera_subscriber')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # Publisher가 RELIABLE이므로 맞춰줌
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # 큐에 쌓이면 지연 생기므로 1로 제한
        )
        self.create_subscription(
            CompressedImage,
            AMR1_IMAGE_TOPIC,
            self.image_callback,
            qos
        )

    def image_callback(self, msg):
        global amr1_latest_frame
        try:
            with amr1_frame_lock:
                # msg.data는 이미 JPEG로 압축된 bytes이므로 재인코딩 없이 그대로 사용
                amr1_latest_frame = bytes(msg.data)
        except Exception as e:
            # 콜백 안에서 예외가 나도 spin() 전체가 죽지 않도록 방지
            print(f"[AMR1 CAM] 콜백 에러: {e}")


def start_ros2_spin():
    """ROS2 노드를 백그라운드 스레드에서 spin"""
    rclpy.init()
    node = Amr1CameraSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Flask 시작 시 ROS2 스레드 한 번만 띄움 (use_reloader=False와 함께 사용해야 중복 실행 안 됨)
ros2_thread = threading.Thread(target=start_ros2_spin, daemon=True)
ros2_thread.start()


# ----------------------------------------
# Detection 카메라 - 방 구석 웹캠
# PC에 직접 연결된 카메라가 아니라, 별도 웹캠이 백엔드 서버로 보낸
# base64 인코딩 프레임을 /api/webcam1/frame/latest/ 에서 받아온다.
# ----------------------------------------
def generate_webcam_frames():
    """백엔드 API에서 base64 프레임을 받아와 MJPEG로 스트리밍"""
    import base64
    import time
    while True:
        try:
            response = requests.get(WEBCAM1_ENDPOINT, timeout=5)
            response.raise_for_status()
            data = response.json()
            frame_b64 = data.get('frame')

            if frame_b64:  # 아직 프레임이 없으면 frame이 null일 수 있음
                frame_bytes = base64.b64decode(frame_b64)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except requests.exceptions.RequestException as e:
            print(f"[WEBCAM1] API 호출 실패: {e}")

        time.sleep(1 / 15)  # 약 15fps로 제한


# ----------------------------------------
# 로그인 / 로그아웃
# ----------------------------------------
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['username'] = username
            flash('로그인 성공!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('로그아웃 되었습니다.', 'info')
    return redirect(url_for('login'))


# ----------------------------------------
# 대시보드 (목차 페이지)
# ----------------------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))

    violations_preview = get_violations()
    disabled_preview = get_disabled_list()

    return render_template(
        'dashboard.html',
        username=session['username'],
        active_page='dashboard',
        violations_data=violations_preview,
        violations_total=len(violations_preview),
        disabled_data=disabled_preview,
        disabled_total=len(disabled_preview),
    )


# ----------------------------------------
# 실시간 카메라 페이지
# ----------------------------------------
@app.route('/cameras')
def cameras():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))
    return render_template('cameras.html', username=session['username'], active_page='cameras')


@app.route('/video_feed1')
def video_feed1():
    # Detection 카메라 (방 구석 웹캠 - 백엔드 API에서 base64 프레임 수신)
    response = Response(generate_webcam_frames(),
                         mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def generate_amr1_frames():
    """ROS2 Subscriber가 받아둔 최신 프레임을 MJPEG로 스트리밍"""
    import time
    while True:
        with amr1_frame_lock:
            frame = amr1_latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 15)  # 약 15fps로 제한 (트래픽/CPU 절감)


@app.route('/video_feed2')
def video_feed2():
    # AMR1 카메라 (ROS2 /compressed 토픽 구독 → 실시간 스트리밍)
    response = Response(generate_amr1_frames(),
                         mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ----------------------------------------
# 불법주차 차량 리스트 페이지
# ----------------------------------------
# ----------------------------------------
# 주차 단속 리스트 확인
# 백엔드 /api/monitor/events/ 호출 → 단속 이벤트(DETECTED~WARNING_ISSUED) 조회
# ----------------------------------------
# ----------------------------------------
# 장애인 차량 관리 페이지
# ----------------------------------------
@app.route('/disabled', methods=['GET', 'POST'])
def disabled():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        plate = request.form.get('plate_number', '').strip()
        if plate:
            success = register_disabled(plate)
            if success:
                flash(f'{plate} 등록되었습니다.', 'success')
            else:
                flash('등록에 실패했습니다. 백엔드 연결을 확인해주세요.', 'danger')
        return redirect(url_for('disabled'))

    data = get_disabled_list()
    return render_template('disabled.html', username=session['username'],
                            data=data, active_page='disabled')


@app.route('/disabled/delete/<plate_number>', methods=['POST'])
def disabled_delete(plate_number):
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))

    success = delete_disabled(plate_number)
    if success:
        flash(f'{plate_number} 삭제되었습니다.', 'info')
    else:
        flash('삭제에 실패했습니다.', 'danger')
    return redirect(url_for('disabled'))


@app.route('/violations')
def violations():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))
    status_filter = request.args.get('status', '')  # 쿼리스트링 ?status=SCANNED 등
    data = get_violations(status_filter if status_filter else None)
    return render_template('violations.html', username=session['username'],
                            data=data, current_status=status_filter, active_page='violations')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)