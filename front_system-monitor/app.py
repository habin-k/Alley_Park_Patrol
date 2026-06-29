from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import requests
import threading

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


AMR1_ENDPOINT = f'{API_BASE_URL}/api/amr1/frame/latest/'
AMR2_ENDPOINT = f'{API_BASE_URL}/api/amr2/frame/latest/'


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

    total_events = len(violations_preview)
    enforced = sum(1 for e in violations_preview if e.get('status') == 'WARNING_ISSUED')
    matched = sum(1 for e in violations_preview if e.get('vehicle_info'))
    # '불법 주차(단속 대상)'는 아직 처리 안 끝난 건수로 집계 (전체 - 단속완료)
    illegal_total = total_events - enforced

    return render_template(
        'dashboard.html',
        username=session['username'],
        active_page='dashboard',
        violations_data=violations_preview,
        violations_total=total_events,
        disabled_data=disabled_preview,
        disabled_total=len(disabled_preview),
        total_events=total_events,
        illegal_total=illegal_total,
        enforced=enforced,
        matched=matched,
    )


# ----------------------------------------
# 웹캠 페이지 (독립 페이지)
# ----------------------------------------
@app.route('/webcam')
def webcam():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))

    detected_events = get_violations(status_filter='DETECTED')

    return render_template(
        'webcam.html',
        username=session['username'],
        active_page='webcam',
        recent_events=detected_events[:10],
        recent_total=len(detected_events),
    )


@app.route('/video_feed_webcam')
def video_feed_webcam():
    # 방 구석 웹캠 - 백엔드 API에서 base64 프레임 수신
    response = Response(generate_webcam_frames(),
                         mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ----------------------------------------
# 실시간 카메라 페이지 (AMR1 + AMR2)
# ----------------------------------------
@app.route('/cameras')
def cameras():
    if 'username' not in session:
        flash('먼저 로그인해주세요.', 'warning')
        return redirect(url_for('login'))
    return render_template('cameras.html', username=session['username'], active_page='cameras')


def generate_amr1_frames():
    import base64, time
    while True:
        try:
            response = requests.get(AMR1_ENDPOINT, timeout=5)
            data = response.json()
            frame_b64 = data.get('frame')
            if frame_b64:
                frame_bytes = base64.b64decode(frame_b64)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[AMR1] API 호출 실패: {e}")
        time.sleep(1 / 15)


def generate_amr2_frames():
    import base64, time
    while True:
        try:
            response = requests.get(AMR2_ENDPOINT, timeout=5)
            data = response.json()
            frame_b64 = data.get('frame')
            if frame_b64:
                frame_bytes = base64.b64decode(frame_b64)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[AMR2] API 호출 실패: {e}")
        time.sleep(1 / 15)


@app.route('/video_feed_amr1')
def video_feed_amr1():
    # AMR1 카메라 (ROS2 /compressed 토픽 구독 → 실시간 스트리밍)
    response = Response(generate_amr1_frames(),
                         mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/video_feed_amr2')
def video_feed_amr2():
    # AMR2 카메라 (ROS2 /compressed 토픽 구독 → 실시간 스트리밍)
    response = Response(generate_amr2_frames(),
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
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1

    PAGE_SIZE = 20
    all_data = get_violations(status_filter if status_filter else None)

    total_count = len(all_data)
    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    if page > total_pages:
        page = total_pages

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_data = all_data[start:end]

    return render_template(
        'violations.html',
        username=session['username'],
        data=page_data,
        current_status=status_filter,
        active_page='violations',
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)