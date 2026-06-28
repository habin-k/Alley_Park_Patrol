# 주차 단속 시스템 - 프로젝트 개요

## 전체 프로세스

### 1단계 - 웹캠 + YOLO (탐지 및 좌표 전송)
웹캠이 주차장을 실시간으로 모니터링하며 YOLO 모델로 차량 객체를 탐지한다. 탐지된 차량의 Nav2 충돌 방지 보정 좌표(`observation_x/y`)를 계산하고, confidence가 높은 순서로 `event_id`를 부여하여 서버로 전송한다. 서버는 이를 `parking_events.vehicle_id` 컬럼에 저장한다. 이 시점에서 `status=DETECTED`로 설정된다. **웹캠은 좌표와 event_id만 전송하며, zone_type과 vehicle_type은 판별하지 않는다.**

### 2단계 - AMR1 (Police 1) - 현장 판별 + 번호판 OCR
AMR1이 서버에서 `observation_x/y` 좌표를 **한 번** 수신한 뒤, Nav2로 **자율적으로 경로를 계획하여** 이동한다. 서버는 좌표를 제공할 뿐 경로 안내를 하지 않는다. 현장 도착 후 두 가지 방법으로 `zone_type`을 판별한다:

- **주황색 주차선 감지 시** → `zone_type=NORMAL` → 정상 주차로 판단하여 **스킵**
- **주황색 주차선 없음** → AMR 팀원이 맵에 사전 정의한 구역 좌표와 로봇의 현재 위치를 비교하여 가장 가까운 구역(`COMPACT` / `DISABLED` / `FIRE` / `EV` / `Not`)을 판별

`vehicle_type=ILLEGAL`로 판단되면 OCR로 번호판 인식 후 `vehicle_info` 테이블에 저장(plate_number, amr_vehicle_x/y, ocr_image_path)하고 `status=SCANNED`으로 업데이트한다. `disabled_vehicle`에 등록된 장애인 차량은 스킵한다.

### 3단계 - AMR2 (Police 2) - 번호판 검증 + 경보
AMR2가 서버에서 `amr_vehicle_x/y` 좌표를 **한 번** 수신한 뒤, Nav2로 **자율적으로 경로를 계획하여** 이동한다. 서버는 좌표를 제공할 뿐 경로 안내를 하지 않는다. 도착 후 YOLO로 번호판 위치를 찾고 OCR 노드에서 번호판 텍스트를 추출하여, DB에 저장된 `plate_number`와 비교한다. 일치하면 `event_id`와 번호판 텍스트를 서버에 전송 → `status=WARNING_ISSUED`로 변경 + 경보 발령. 불일치하면 서버에 `match=false` + `event_id` 전송 → 해당 이벤트 삭제. 좌표가 없으면 제자리로 복귀한다.

## status 흐름

```
DETECTED → SCANNED → WARNING_ISSUED
 (웹캠)    (AMR1)     (AMR2, 경보)
```

## 노드별 역할

| 노드 | 역할 | DB 작업 |
|------|------|---------|
| **웹캠 + YOLO** | 차량 탐지, bbox 좌표 계산 | `parking_events` INSERT (`vehicle_id`, `observation_x/y`, `status=DETECTED`) |
| **AMR1 (Police 1)** | 주차선 색 판별 + 맵 좌표 비교로 구역 판단 (AMR 내부), 번호판 OCR | `vehicle_info` INSERT (`plate_number`, `ocr_image`) + `parking_events.status=SCANNED` |
| **AMR2 (Police 2)** | YOLO로 번호판 탐지, OCR 텍스트 추출 후 구역별 검증, 경보 발령 | 정상 → 이벤트 삭제 / 불법 → `status=WARNING_ISSUED` + base64 이미지 수신 |

## 주차 구역 종류 (zone_type)

| 값 | 판별 방법 | 설명 | 처리 |
|----|----------|------|------|
| `NORMAL` | 주황색 주차선 감지 | 정상 주차구역 | **스킵** |
| `COMPACT` | 맵 좌표 비교 | 경차 전용 (AMR팀 사전 정의) | 불법 처리 |
| `DISABLED` | 맵 좌표 비교 | 장애인 전용 (AMR팀 사전 정의) | `disabled_vehicle` 조회 후 판단 |
| `FIRE` | 맵 좌표 비교 | 소방차 전용 (AMR팀 사전 정의) | 항상 불법 |
| `EV` | 맵 좌표 비교 | 전기차 전용 (AMR팀 사전 정의) | 불법 처리 |
| `Not` | 주차선 없음 | 주차 구역 아님 | 불법 처리 |

> **zone_type 판별 로직 (AMR1 기준)**
> 1. 주황색 주차선 감지 → `NORMAL` → 스킵
> 2. 주황색 아님 → 로봇 현재 위치와 맵에 사전 정의된 구역 좌표 비교 → 가장 가까운 구역으로 판별
> 3. 어떤 구역에도 해당 없음 → `Not`

## 기술 스택

- **백엔드**: Django 5.2.15
- **DB**: Supabase (PostgreSQL) - 클라우드
- **API 문서**: Swagger UI (`http://<서버IP>:8000/api/schema/swagger-ui/`)
- **시스템 모니터**: `http://<서버IP>:8000/monitor/` (로그인: user / password)
- **객체 탐지**: YOLOv8 (ultralytics), 학습 모델: `/home/rokey/detect_model/total.pt`
- **가상환경**: `/home/rokey/kimandreas/to_students/day5/venv/`

## 중앙 서버

- IP: `192.168.107.42`
- 포트: `8000`
- 실행: `python manage.py runserver 0.0.0.0:8000` (단축키: `run`)

## DB 테이블

### parking_events
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | int | PK, Auto Increment (DB 내부용) |
| `vehicle_id` | int | 웹캠이 `event_id` 이름으로 전송한 차량 식별자 (nullable). AMR 간 차량 추적에 사용 |
| `observation_x` | float | Nav2 안전 접근 보정 x 좌표 |
| `observation_y` | float | Nav2 안전 접근 보정 y 좌표 |
| `status` | varchar | `DETECTED` / `SCANNED` / `WARNING_ISSUED` (인덱스, 기본값 DETECTED) |
| `created_at` | timestamp | 최초 웹캠 감지 시각 (자동) |

### vehicle_info
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | int | PK, Auto Increment (DB 내부용) |
| `event_id` | int (FK) | parking_events.id 참조 (CASCADE, related_name='vehicle_info') |
| `plate_number` | varchar | AMR1 OCR 인식 번호판 텍스트 |
| `ocr_image` | text | AMR1 근접 촬영 번호판 이미지 (base64 인코딩, nullable) |
| `updated_at` | timestamp | 마지막 수정 시각 (자동) |

### disabled_vehicle
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | int | PK, Auto Increment |
| `plate_number` | varchar | 사전 등록된 장애인 차량 번호판 (UNIQUE, 관리자 직접 입력) |
| `registered_at` | timestamp | 등록 시각 (자동) |

## API 엔드포인트

### 웹캠 노드
| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/parking/` | 차량 탐지 이벤트 생성 (status=DETECTED) |
| GET | `/api/parking/list/` | 전체 이벤트 조회 |
| DELETE | `/api/parking/<id>/delete/` | DETECTED 이벤트 삭제 (차량 이동 시) |

### AMR1 (Police 1)
| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/parking/next/` | DETECTED 이벤트의 `vehicle_id` + 좌표를 **한 번** 수신 → AMR1이 Nav2로 자율 경로 계획 |
| GET | `/api/disabled/<번호판>/` | 장애인 차량 여부 확인 |
| POST | `/api/vehicle/` | 번호판(`plate_number`) + base64 이미지(`ocr_image`) 저장 + status → SCANNED |

### AMR2 (Police 2)
| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/vehicle/next/` | SCANNED 이벤트의 좌표를 **한 번** 수신 → AMR2가 Nav2로 자율 경로 계획. null이면 복귀 |
| POST | `/api/vehicle/verify/` | 번호판 매칭 결과 전송 (match=true → WARNING_ISSUED, match=false → 이벤트 삭제) |

### 모니터링
| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/monitor/summary/` | 전체 현황 통계 |
| GET | `/api/monitor/events/` | 이벤트 목록 (vehicle_info 포함, ?status= 필터) |
| GET | `/api/monitor/disabled/` | 장애인 차량 목록 |
| POST | `/api/monitor/disabled/register/` | 장애인 차량 등록 |
| DELETE | `/api/monitor/disabled/<번호판>/` | 장애인 차량 삭제 |
| GET | `/api/monitor/plate-match/` | AMR1/2 번호판 매칭 결과 |
| POST | `/api/webcam1/frame/` | 웹캠1 프레임 전송 (base64) |
| GET | `/api/webcam1/frame/latest/` | 웹캠1 최신 프레임 수신 |
| GET | `/api/webcam1/stream/` | 웹캠1 MJPEG 스트리밍 |
| POST | `/api/webcam2/frame/` | 웹캠2 프레임 전송 (base64) |
| GET | `/api/webcam2/frame/latest/` | 웹캠2 최신 프레임 수신 |
| GET | `/api/webcam2/stream/` | 웹캠2 MJPEG 스트리밍 |
| POST | `/api/amr1/frame/` | AMR1 프레임 전송 (base64) |
| GET | `/api/amr1/frame/latest/` | AMR1 최신 프레임 수신 |
| POST | `/api/amr2/frame/` | AMR2 프레임 전송 (base64) |
| GET | `/api/amr2/frame/latest/` | AMR2 최신 프레임 수신 |

## 개발 명령어 단축키

```bash
mkm   # python manage.py makemigrations
mig   # python manage.py migrate
run   # python manage.py runserver 0.0.0.0:8000
```

## 환경변수 (.env)

```
SECRET_KEY=...
DB_NAME=postgres
DB_USER=postgres.<프로젝트ID>
DB_PASSWORD=...
DB_HOST=aws-0-ap-northeast-2.pooler.supabase.com
DB_PORT=6543
```

`.env`는 `.gitignore`에 포함되어 있어 git에 올라가지 않음. 팀원에게 직접 전달 필요.
