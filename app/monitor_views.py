import base64
import threading
import time

from django.http import HttpResponse
from django.http import StreamingHttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter, inline_serializer
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers
from .monitor_dashboard import VALID_USERNAME, VALID_PASSWORD

# 서버 메모리에 최신 프레임 저장 (DB 저장 안 함)
_webcam1_store = {'frame': None, 'updated_at': None}
_webcam2_store = {'frame': None, 'updated_at': None}
_amr1_store    = {'frame': None, 'updated_at': None}
_amr2_store    = {'frame': None, 'updated_at': None}
_webcam_store  = _webcam1_store  # 기존 코드 호환
_lock = threading.Lock()
from django.db.models import Count, Q
from .models import ParkingEvent, VehicleInfo, DisabledVehicle
from .serializers import DisabledVehicleSerializer


# ── 0. 로그인 / 로그아웃 ─────────────────────────────────────

@extend_schema(
    summary="[인증] 로그인",
    description="username과 password를 전송하면 세션을 생성합니다.\n\n기본 계정: `user` / `password`",
    request=inline_serializer(
        name='LoginRequest',
        fields={
            'username': serializers.CharField(),
            'password': serializers.CharField(),
        }
    ),
    responses={
        200: inline_serializer(
            name='LoginSuccess',
            fields={'success': serializers.BooleanField(), 'username': serializers.CharField()}
        ),
        401: inline_serializer(
            name='LoginFail',
            fields={'success': serializers.BooleanField(), 'error': serializers.CharField()}
        ),
    },
)
@api_view(['POST'])
def api_login(request):
    username = request.data.get('username', '')
    password = request.data.get('password', '')
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        request.session['username'] = username
        return Response({'success': True, 'username': username})
    return Response(
        {'success': False, 'error': '아이디 또는 비밀번호가 올바르지 않습니다.'},
        status=status.HTTP_401_UNAUTHORIZED,
    )


@extend_schema(
    summary="[인증] 로그아웃",
    description="현재 세션을 삭제합니다.",
    responses={200: inline_serializer(
        name='LogoutSuccess',
        fields={'success': serializers.BooleanField()}
    )},
)
@api_view(['POST'])
def api_logout(request):
    request.session.flush()
    return Response({'success': True})


# ── 1. 대시보드 요약 ──────────────────────────────────────────

@extend_schema(
    summary="[모니터] 전체 현황 요약",
    description="status별, zone별 이벤트 수를 반환. 대시보드 상단 통계 카드용.",
)
@api_view(['GET'])
def summary(request):
    total = ParkingEvent.objects.count()
    by_status = {
        item['status']: item['count']
        for item in ParkingEvent.objects.values('status').annotate(count=Count('id'))
    }
    illegal_total = ParkingEvent.objects.filter(
        status__in=['SCANNED', 'WARNING_ISSUED']
    ).count()
    enforced = ParkingEvent.objects.filter(status='WARNING_ISSUED').count()

    return Response({
        'total_events': total,
        'illegal_total': illegal_total,
        'enforced': enforced,
        'by_status': by_status,
    })


# ── 2. 이벤트 목록 (vehicle_info 포함) ───────────────────────

@extend_schema(
    summary="[모니터] 주차 이벤트 목록",
    description="parking_events + vehicle_info 조인 데이터. status 파라미터로 필터 가능. 예: ?status=SCANNED",
    parameters=[
        OpenApiParameter('status', OpenApiTypes.STR, description='DETECTED / SCANNED / VERIFIED / WARNING_ISSUED'),
    ],
)
@api_view(['GET'])
def event_list(request):
    qs = ParkingEvent.objects.all().order_by('-created_at')
    status_filter = request.query_params.get('status')
    if status_filter:
        qs = qs.filter(status=status_filter)

    data = []
    for event in qs:
        item = {
            'event_id':      event.id,
            'observation_x': event.observation_x,
            'observation_y': event.observation_y,
            'status':        event.status,
            'created_at':    event.created_at,
            'vehicle_info':  None,
        }
        try:
            vi = event.vehicle_info
            item['vehicle_info'] = {
                'plate_number': vi.plate_number,
                'ocr_image':    vi.ocr_image,
            }
        except VehicleInfo.DoesNotExist:
            pass
        data.append(item)

    return Response(data)


# ── 3. 장애인 차량 목록 / 등록 / 삭제 ────────────────────────

@extend_schema(
    summary="[모니터] 장애인 차량 목록 조회",
    responses={200: DisabledVehicleSerializer(many=True)},
)
@api_view(['GET'])
def disabled_list(request):
    vehicles = DisabledVehicle.objects.all().order_by('-registered_at')
    serializer = DisabledVehicleSerializer(vehicles, many=True)
    return Response(serializer.data)


@extend_schema(
    summary="[모니터] 장애인 차량 등록",
    description="번호판을 disabled_vehicle 테이블에 등록.",
    request=DisabledVehicleSerializer,
    responses={201: DisabledVehicleSerializer},
)
@api_view(['POST'])
def disabled_register(request):
    plate = request.data.get('plate_number', '').strip()
    if not plate:
        return Response({'error': 'plate_number is required'}, status=status.HTTP_400_BAD_REQUEST)
    obj, created = DisabledVehicle.objects.get_or_create(plate_number=plate)
    if not created:
        return Response({'error': '이미 등록된 번호판입니다.'}, status=status.HTTP_409_CONFLICT)
    return Response({'plate_number': obj.plate_number, 'registered_at': obj.registered_at}, status=status.HTTP_201_CREATED)


@extend_schema(
    summary="[모니터] 장애인 차량 삭제",
    description="번호판으로 disabled_vehicle 테이블에서 삭제.",
)
@api_view(['DELETE'])
def disabled_delete(request, plate_number):
    try:
        obj = DisabledVehicle.objects.get(plate_number=plate_number)
        obj.delete()
        return Response({'message': f'{plate_number} 삭제 완료'})
    except DisabledVehicle.DoesNotExist:
        return Response({'error': '등록되지 않은 번호판입니다.'}, status=status.HTTP_404_NOT_FOUND)


# ── 4. 번호판 비교 결과 (AMR1 vs AMR2) ───────────────────────

@extend_schema(
    summary="[모니터] AMR1/2 번호판 비교 결과",
    description="VERIFIED 이상 상태의 이벤트에서 AMR1(vehicle_info.plate_number)과 AMR2 매칭 결과를 반환.",
)
@api_view(['GET'])
def plate_match_result(request):
    events = ParkingEvent.objects.filter(
        status='WARNING_ISSUED'
    ).order_by('-created_at')

    data = []
    for event in events:
        try:
            vi = event.vehicle_info
            data.append({
                'event_id':     event.id,
                'amr1_plate':   vi.plate_number,
                'match_result': 'MATCHED',
                'status':       event.status,
                'created_at':   event.created_at,
            })
        except VehicleInfo.DoesNotExist:
            pass

    return Response(data)


# ── 6. 웹캠 프레임 전송 / 수신 ───────────────────────────────

@extend_schema(
    summary="[웹캠 PC] 최신 프레임 전송",
    description="웹캠 PC가 base64로 인코딩한 프레임을 0.5초마다 전송. DB 저장 없이 서버 메모리에만 유지.",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'frame': {'type': 'string', 'description': 'base64 인코딩된 JPEG 이미지'},
            },
            'required': ['frame'],
        }
    },
)
@api_view(['POST'])
def webcam_frame_upload(request):
    frame = request.data.get('frame')
    if not frame:
        return Response({'error': 'frame is required'}, status=status.HTTP_400_BAD_REQUEST)

    from django.utils import timezone
    with _lock:
        _webcam_store['frame'] = frame
        _webcam_store['updated_at'] = timezone.now().isoformat()

    return Response({'status': 'ok'})


@extend_schema(
    summary="[모니터] 최신 웹캠 프레임 수신 (base64)",
    description="최신 프레임을 base64로 반환. 프레임이 없으면 frame=null.",
)
@api_view(['GET'])
def webcam_frame_get(request):
    with _lock:
        return Response({
            'frame': _webcam_store['frame'],
            'updated_at': _webcam_store['updated_at'],
        })


def webcam_image(request):
    """단일 이미지 반환 (정적)"""
    with _lock:
        frame_b64 = _webcam_store['frame']
    if not frame_b64:
        return HttpResponse('No frame yet', status=503)
    image_bytes = base64.b64decode(frame_b64)
    return HttpResponse(image_bytes, content_type='image/jpeg')


def _mjpeg_generator():
    while True:
        with _lock:
            frame_b64 = _webcam_store['frame']
        if frame_b64:
            image_bytes = base64.b64decode(frame_b64)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
            )
        time.sleep(0.1)


def webcam_stream(request):
    """실시간 MJPEG 스트리밍. 브라우저에서 자동 갱신."""
    return StreamingHttpResponse(
        _mjpeg_generator(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ── 6-2. 웹캠2 프레임 전송 / 수신 ───────────────────────────

@api_view(['POST'])
def webcam2_frame_upload(request):
    frame = request.data.get('frame')
    if not frame:
        return Response({'error': 'frame is required'}, status=status.HTTP_400_BAD_REQUEST)
    from django.utils import timezone
    with _lock:
        _webcam2_store['frame'] = frame
        _webcam2_store['updated_at'] = timezone.now().isoformat()
    return Response({'status': 'ok'})


@api_view(['GET'])
def webcam2_frame_get(request):
    with _lock:
        return Response({
            'frame': _webcam2_store['frame'],
            'updated_at': _webcam2_store['updated_at'],
        })


def _mjpeg_generator2():
    while True:
        with _lock:
            frame_b64 = _webcam2_store['frame']
        if frame_b64:
            image_bytes = base64.b64decode(frame_b64)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
            )
        time.sleep(0.1)


def webcam2_stream(request):
    return StreamingHttpResponse(
        _mjpeg_generator2(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ── 7. AMR1 프레임 전송 / 수신 ───────────────────────────────

@extend_schema(
    summary="[AMR1] 최신 프레임 전송",
    description="AMR1이 base64로 인코딩한 프레임을 전송. DB 저장 없이 서버 메모리에만 유지.",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'frame': {'type': 'string', 'description': 'base64 인코딩된 JPEG 이미지'},
            },
            'required': ['frame'],
        }
    },
)
@api_view(['POST'])
def amr1_frame_upload(request):
    frame = request.data.get('frame')
    if not frame:
        return Response({'error': 'frame is required'}, status=status.HTTP_400_BAD_REQUEST)

    from django.utils import timezone
    with _lock:
        _amr1_store['frame'] = frame
        _amr1_store['updated_at'] = timezone.now().isoformat()

    return Response({'status': 'ok'})


@extend_schema(
    summary="[모니터] 최신 AMR1 프레임 수신 (base64)",
    description="최신 프레임을 base64로 반환. 프레임이 없으면 frame=null.",
)
@api_view(['GET'])
def amr1_frame_get(request):
    with _lock:
        return Response({
            'frame': _amr1_store['frame'],
            'updated_at': _amr1_store['updated_at'],
        })


# ── 8. AMR2 프레임 전송 / 수신 ───────────────────────────────

@extend_schema(
    summary="[AMR2] 최신 프레임 전송",
    description="AMR2가 base64로 인코딩한 프레임을 전송. DB 저장 없이 서버 메모리에만 유지.",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'frame': {'type': 'string', 'description': 'base64 인코딩된 JPEG 이미지'},
            },
            'required': ['frame'],
        }
    },
)
@api_view(['POST'])
def amr2_frame_upload(request):
    frame = request.data.get('frame')
    if not frame:
        return Response({'error': 'frame is required'}, status=status.HTTP_400_BAD_REQUEST)

    from django.utils import timezone
    with _lock:
        _amr2_store['frame'] = frame
        _amr2_store['updated_at'] = timezone.now().isoformat()

    return Response({'status': 'ok'})


@extend_schema(
    summary="[모니터] 최신 AMR2 프레임 수신 (base64)",
    description="최신 프레임을 base64로 반환. 프레임이 없으면 frame=null.",
)
@api_view(['GET'])
def amr2_frame_get(request):
    with _lock:
        return Response({
            'frame': _amr2_store['frame'],
            'updated_at': _amr2_store['updated_at'],
        })
