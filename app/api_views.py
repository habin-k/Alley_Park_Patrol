from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema
from django.db import connection
from .models import ParkingEvent, VehicleInfo, DisabledVehicle
from .serializers import (
    ParkingEventCreateSerializer, ParkingEventSerializer,
    ParkingEventNextSerializer,
    VehicleInfoCreateSerializer, VehicleInfoNextSerializer,
    DisabledVehicleSerializer,
)


# ── 웹캠 노드 ─────────────────────────────────────────────────

@extend_schema(
    summary="[웹캠] 차량 탐지 이벤트 생성",
    description="YOLO가 차량을 탐지하면 호출. parking_events 테이블에 저장되며 status=DETECTED로 시작.",
    request=ParkingEventCreateSerializer,
    responses={201: ParkingEventSerializer},
)
@api_view(['POST'])
def parking_create(request):
    serializer = ParkingEventCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    event = ParkingEvent.objects.create(
        vehicle_id=serializer.validated_data.get('event_id'),
        observation_x=serializer.validated_data['observation_x'],
        observation_y=serializer.validated_data['observation_y'],
        status='DETECTED',
    )
    return Response({'status': 'ok', 'event_id': event.id}, status=status.HTTP_201_CREATED)


@extend_schema(
    summary="[공통] 전체 주차 이벤트 조회",
    description="parking_events 테이블 전체 조회. 최신순 정렬.",
    responses={200: ParkingEventSerializer(many=True)},
)
@api_view(['GET'])
def parking_list(request):
    events = ParkingEvent.objects.all().order_by('-created_at')
    serializer = ParkingEventSerializer(events, many=True)
    return Response(serializer.data)


@extend_schema(
    summary="[웹캠] 차량 이동 시 기존 탐지 기록 삭제",
    description="차량이 이동하여 좌표가 바뀐 경우, 기존 DETECTED 상태 이벤트를 삭제. DETECTED 상태가 아니면 삭제 불가.",
)
@api_view(['DELETE'])
def parking_delete(request, event_id):
    try:
        event = ParkingEvent.objects.get(id=event_id)
    except ParkingEvent.DoesNotExist:
        return Response({'error': 'event not found'}, status=status.HTTP_404_NOT_FOUND)
    if event.status != 'DETECTED':
        return Response({'error': f'DETECTED 상태만 삭제 가능합니다. 현재 상태: {event.status}'}, status=status.HTTP_400_BAD_REQUEST)
    event.delete()
    return Response({'status': 'deleted', 'event_id': event_id})


# ── AMR1 노드 ─────────────────────────────────────────────────

@extend_schema(
    summary="[AMR1] 목표 좌표 수신",
    description="AMR1이 한 번 호출하여 목표 좌표를 받고 Nav2로 스스로 경로를 계획함. "
                "status=DETECTED 중 가장 오래된 이벤트의 event_id와 observation_x/y를 반환. "
                "없으면 event=null 반환 → AMR1 대기.",
    responses={200: ParkingEventNextSerializer},
)
@api_view(['GET'])
def parking_next(request):
    event = ParkingEvent.objects.filter(status='DETECTED').order_by('created_at').first()
    if not event:
        return Response({'event': None})
    return Response({
        'event_id':      event.id,
        'observation_x': event.observation_x,
        'observation_y': event.observation_y,
        'created_at':    event.created_at,
    })


@extend_schema(
    summary="[AMR1] 번호판 정보 저장",
    description="AMR1이 OCR로 번호판 인식 후 호출. vehicle_info 저장 + 해당 이벤트 status → SCANNED.",
    request=VehicleInfoCreateSerializer,
    responses={201: VehicleInfoNextSerializer},
)
@api_view(['POST'])
def vehicle_create(request):
    serializer = VehicleInfoCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    try:
        event = ParkingEvent.objects.get(id=data['event_id'])
    except ParkingEvent.DoesNotExist:
        return Response({'error': 'event not found'}, status=status.HTTP_404_NOT_FOUND)

    vehicle_info = VehicleInfo.objects.create(
        event=event,
        plate_number=data['plate_number'],
        ocr_image=data.get('ocr_image'),
    )
    event.status = 'SCANNED'
    event.save()

    return Response({'vehicle_info_id': vehicle_info.id}, status=status.HTTP_201_CREATED)


@extend_schema(
    summary="[AMR1] 장애인 차량 여부 확인",
    description="번호판으로 disabled_vehicle 테이블 조회. 등록된 장애인 차량이면 is_disabled=true.",
    responses={200: DisabledVehicleSerializer},
)
@api_view(['GET'])
def disabled_check(request, plate_number):
    is_disabled = DisabledVehicle.objects.filter(plate_number=plate_number).exists()
    return Response({'plate_number': plate_number, 'is_disabled': is_disabled})


# ── AMR2 노드 ─────────────────────────────────────────────────

@extend_schema(
    summary="[OCR 브리지] vehicle_id로 parking_events 조회",
    description="OCR 노드가 사용하는 vehicle_id(웹캠 confidence 순 번호)로 parking_events.id(DB PK)를 조회. 브리지가 API 호출 시 DB PK로 변환하는 데 사용.",
)
@api_view(['GET'])
def parking_by_vehicle(request, vehicle_id):
    event = ParkingEvent.objects.filter(vehicle_id=vehicle_id).order_by('-created_at').first()
    if not event:
        return Response({'error': 'not found'}, status=status.HTTP_404_NOT_FOUND)
    return Response({'id': event.id, 'vehicle_id': event.vehicle_id, 'status': event.status})


@extend_schema(
    summary="[OCR 브리지] DB event_id로 번호판 조회",
    description="parking_events.id(DB PK)로 vehicle_info의 plate_number 조회. 브리지가 servertoocr 토픽으로 재발행함.",
)
@api_view(['GET'])
def vehicle_get(request, event_id):
    try:
        vi = VehicleInfo.objects.get(event_id=event_id)
    except VehicleInfo.DoesNotExist:
        return Response({'error': 'not found'}, status=status.HTTP_404_NOT_FOUND)
    return Response({'event_id': event_id, 'plate_number': vi.plate_number})


@extend_schema(
    summary="[AMR2] 목표 좌표 수신",
    description="AMR2가 한 번 호출하여 목표 좌표를 받고 Nav2로 스스로 경로를 계획함. "
                "status=SCANNED 중 가장 오래된 vehicle_info의 event_id, plate_number를 반환. "
                "없으면 vehicle_info=null 반환 → AMR2 제자리 복귀.",
    responses={200: VehicleInfoNextSerializer},
)
@api_view(['GET'])
def vehicle_next(request):
    vehicle_info = VehicleInfo.objects.filter(
        event__status='SCANNED',
    ).order_by('event__created_at').first()

    if not vehicle_info:
        return Response({'vehicle_info': None})

    serializer = VehicleInfoNextSerializer(vehicle_info)
    return Response(serializer.data)


@extend_schema(
    summary="[AMR2] 번호판 검증 결과 전송",
    description=(
        "AMR2 OCR로 번호판 재인식 후 비교 결과 전송.\n"
        "- match=true: status → WARNING_ISSUED, ocr_image(base64) 저장\n"
        "- match=false: parking_events + vehicle_info 레코드 삭제"
    ),
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'event_id':     {'type': 'integer'},
                'match':        {'type': 'boolean'},
                'plate_number': {'type': 'string', 'description': '일치한 경우 번호판 텍스트'},
                'ocr_image':    {'type': 'string', 'description': 'base64 인코딩된 번호판 이미지 (선택)'},
            },
            'required': ['event_id', 'match'],
        }
    },
)
@api_view(['POST'])
def vehicle_verify(request):
    event_id = request.data.get('event_id')
    match    = request.data.get('match')

    if event_id is None or match is None:
        return Response({'error': 'event_id와 match는 필수입니다.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        event = ParkingEvent.objects.get(id=event_id)
    except ParkingEvent.DoesNotExist:
        return Response({'error': 'event not found'}, status=status.HTTP_404_NOT_FOUND)

    if match:
        event.status = 'WARNING_ISSUED'
        event.save()
        ocr_image = request.data.get('ocr_image')
        if ocr_image:
            try:
                vi = event.vehicle_info
                vi.ocr_image = ocr_image
                vi.save()
            except VehicleInfo.DoesNotExist:
                pass
        return Response({'result': 'matched', 'event_id': event.id, 'status': event.status})
    else:
        event.delete()
        return Response({'result': 'not_matched', 'event_id': event_id, 'deleted': True})
