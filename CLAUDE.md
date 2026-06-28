# 주차 단속 시스템 - 프로젝트 개요

## 전체 프로세스

### 1단계 - 웹캠 + YOLO (탐지 및 좌표 전송)
웹캠이 주차장을 실시간으로 모니터링하며 YOLO 모델로 차량 객체를 탐지한다. 탐지된 차량의 Nav2 충돌 방지 보정 좌표(`observation_x/y`)를 계산하고, confidence가 높은 순서로 `vehicle_id`를 부여하여 `parking_events` 테이블에 저장한다. 이 시점에서 `status=DETECTED`로 설정된다. **웹캠은 좌표와 vehicle_id만 전송하며, zone_type과 vehicle_type은 판별하지 않는다. zone_type/vehicle_type은 DB에 저장되지 않고 AMR 내부에서만 사용한다.**

브리지(`bridge_webcam.py`)는 이전 프레임과 현재 프레임을 **diff 비교**하여 처리한다:
- **새로 생긴 차량** → `POST /api/parking/` → 서버에서 받은 `event_id` 저장
- **없어진 차량** → `DELETE /api/parking/<event_id>/delete/`
- **그대로인 차량** → 아무것도 안 함

### 2단계 - AMR1 (Police 1) - 현장 판별 + 번호판 OCR
AMR1이 **웹캠 노드로부터 ROS2 토픽으로** `observation_x/y` 좌표와 `vehicle_id`를 직접 수신한 뒤, Nav2로 **자율적으로 경로를 계획하여** 이동한다. **서버와 직접 통신하지 않는다.** 현장 도착 후 두 가지 방법으로 `zone_type`을 판별한다:

- **주황색 주차선 감지 시** → `zone_type=NORMAL` → 정상 주차로 판단하여 **스킵**
- **주황색 주차선 없음** → AMR 팀원이 맵에 사전 정의한 구역 좌표와 로봇의 현재 위치를 비교하여 가장 가까운 구역(`DISABLED` / `FIRE` / `Not`)을 판별

불법 주차로 판단되면 OCR로 번호판 인식 후 `regist_car` ROS2 토픽으로 발행한다. `bridge_ocr.py`가 이를 수신하여 서버 `POST /api/vehicle/`를 호출하고, `vehicle_info` 테이블에 저장(`plate_number`, `ocr_image`)하며 `status=SCANNED`으로 업데이트한다. `disabled_vehicle`에 등록된 장애인 차량은 스킵한다. **zone_type은 DB에 저장하지 않으며 AMR 내부에서만 사용한다.**

### 3단계 - AMR2 (Police 2) - 번호판 검증 + 경보
AMR2가 **웹캠 노드로부터 ROS2 토픽으로** `observation_x/y` 좌표와 `vehicle_id`를 직접 수신한 뒤, Nav2로 **자율적으로 경로를 계획하여** 이동한다. **서버와 직접 통신하지 않으며**, 모든 서버 통신은 `bridge_ocr.py`가 담당한다. 좌표가 없으면 제자리로 복귀한다.

도착 후 zone_type에 따라 처리 방식이 다르다. **주차 구역 정보는 DB에 저장되지 않으며, AMR 내부에 사전 정의되어 있다.**

#### DISABLED (장애인 전용) 구역
1. YOLO + OCR로 번호판 텍스트 추출
2. `disabled` ROS2 토픽으로 번호판 발행 → `bridge_ocr.py`가 `GET /api/disabled/<plate>/` 호출
3. `disabled_result` 토픽(Bool)으로 결과 수신
   - `True` (장애인 차량, 정상) → 스킵
   - `False` (불법) → `disabled_result_id` 토픽으로 `vehicle_id` + base64 이미지 발행 → `bridge_ocr.py`가 `POST /api/vehicle/verify/ match=true` 호출 → `status=WARNING_ISSUED`

#### FIRE (소방차 전용) 구역
- 소방차는 전용 번호판을 사용하므로 번호판으로 소방차 여부를 판별한다
1. YOLO + OCR로 번호판 텍스트 추출
2. `request_car` 토픽으로 `vehicle_id` 발행 → `bridge_ocr.py`가 서버에서 AMR1이 저장한 `plate_number` 조회 → `servertoocr` 토픽으로 수신하여 plate-match 수행
   - 일치 (실제 소방차) → `firecar_result(True)` 토픽 발행 → `bridge_ocr.py`가 `POST /api/vehicle/verify/ match=false` 호출 → 이벤트 삭제
   - 불일치 (소방차 아님) → `firecar_result_id` 토픽으로 `vehicle_id` + base64 이미지 발행 → `bridge_ocr.py`가 `POST /api/vehicle/verify/ match=true` 호출 → `status=WARNING_ISSUED`

#### 그 외 구역 (Not)
YOLO로 번호판 위치를 찾고 OCR로 번호판 텍스트를 추출하여 DB에 저장된 `plate_number`와 비교한다. 일치하면 `match_result_id` 토픽 발행 → `bridge_ocr.py`가 `POST /api/vehicle/verify/ match=true` 호출 → `status=WARNING_ISSUED`. 불일치하면 `match_result(False)` 토픽 발행 → `bridge_ocr.py`가 `POST /api/vehicle/verify/ match=false` 호출 → 이벤트 삭제.

## status 흐름

```
DETECTED → SCANNED → WARNING_ISSUED
 (웹캠)    (AMR1)     (AMR2, 경보)
```

## 노드별 역할

| 노드 | 역할 | DB 작업 |
|------|------|---------|
| **웹캠 + YOLO** | 차량 탐지, bbox 좌표 계산 | `parking_events` INSERT (`status=DETECTED`) |
| **AMR1 (Police 1)** | 웹캠 ROS2 토픽으로 좌표 수신 → Nav2 자율 이동 → 주차선 색 판별 + 맵 좌표 비교로 구역 판단 → 번호판 OCR 후 `regist_car` 토픽 발행. **서버 직접 통신 없음** | 직접 DB 접근 없음 (bridge_ocr 경유) |
| **AMR2 (Police 2)** | 웹캠 ROS2 토픽으로 좌표 수신 → Nav2 자율 이동 → YOLO 번호판 탐지 + OCR → 구역별 검증 후 결과 토픽 발행. **서버 직접 통신 없음** | 직접 DB 접근 없음 (bridge_ocr 경유) |
| **bridge_ocr.py** | OCR 노드(plate_ocr_node.py) ↔ Django 서버 중계. vehicle_id → DB id 변환 담당 | 직접 DB 접근 없음 (HTTP API 경유) |

## 주차 구역 종류 (zone_type)

| 값 | 판별 방법 | 설명 | 처리 |
|----|----------|------|------|
| `NORMAL` | 주황색 주차선 감지 | 정상 주차구역 | **스킵** |
| `DISABLED` | 맵 좌표 비교 | 장애인 전용 (AMR팀 사전 정의) | `disabled_vehicle` 조회 후 판단 |
| `FIRE` | 맵 좌표 비교 | 소방차 전용 (AMR팀 사전 정의) | 번호판 plate-match → 소방차면 정상, 아니면 불법 |
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
| `vehicle_id` | int | 웹캠이 confidence 높은 순으로 부여하는 차량 식별자. AMR1/AMR2가 차량 구분에 사용 (nullable) |
| `observation_x` | float | Nav2 안전 접근 보정 x 좌표 |
| `observation_y` | float | Nav2 안전 접근 보정 y 좌표 |
| `status` | varchar | `DETECTED` / `SCANNED` / `WARNING_ISSUED` (인덱스, 기본값 DETECTED) |
| `created_at` | timestamp | 최초 웹캠 감지 시각 (자동) |

> **⚠️ event_id 이름 혼용 주의 — 매우 헷갈리는 포인트**
>
> 시스템에서 `event_id`라는 이름이 두 가지 **전혀 다른 값**으로 사용된다.
>
> | 사용 위치 | `event_id`가 가리키는 값 | 실제 컬럼 |
> |-----------|--------------------------|-----------|
> | ROS2 통신 (AMR1 ↔ AMR2 ↔ OCR ↔ bridge_ocr) | 웹캠이 confidence 순으로 부여한 차량 식별자 | `parking_events.vehicle_id` |
> | `vehicle_info` 테이블의 `event_id` 컬럼 | Django 자동 생성 DB PK | `parking_events.id` |
>
> **즉, ROS2 토픽에서 주고받는 `event_id`는 `parking_events.vehicle_id` 값이고, DB의 `vehicle_info.event_id`는 `parking_events.id`(PK) 값이다. 이름은 같지만 완전히 다른 값이다.**
>
> `bridge_ocr.py`가 이 변환을 담당한다: ROS2에서 `event_id`(=vehicle_id) 수신 → `GET /api/parking/by-vehicle/<vehicle_id>/`로 DB PK 조회 → DB PK를 사용해 API 호출.
>
> - `parking_events.id`: Django 자동 생성 PK. DB 관계 유지용 (vehicle_info FK 등). bridge_ocr가 변환 후 API 호출에 사용.
> - `parking_events.vehicle_id`: 웹캠이 부여하는 운영 식별자. ROS2 통신에서 `event_id`라는 이름으로 사용.

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
> AMR1/AMR2는 서버와 직접 통신하지 않음. 웹캠 노드로부터 좌표와 vehicle_id를 직접 수신하고, 서버 통신은 bridge_ocr.py가 담당.

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/disabled/<번호판>/` | 장애인 차량 여부 확인 |
| POST | `/api/vehicle/` | 번호판(`plate_number`) + base64 이미지(`ocr_image`) 저장 + status → SCANNED |

### AMR2 (Police 2)
| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/vehicle/verify/` | 번호판 매칭 결과 전송 (match=true → WARNING_ISSUED, match=false → 이벤트 삭제) |

### bridge_ocr.py (OCR 노드 ↔ 서버 중계)
| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/parking/by-vehicle/<vehicle_id>/` | vehicle_id → parking_events.id(DB PK) 변환 조회 |
| GET | `/api/vehicle/<db_id>/` | AMR1이 저장한 plate_number 조회 (AMR2 매칭용) |

> **vehicle_id → DB id 변환이 필요한 이유**
> OCR 노드(plate_ocr_node.py)는 웹캠이 부여한 `vehicle_id`로 차량을 식별하지만, Django API는 `parking_events.id`(DB PK)를 기준으로 동작한다. bridge_ocr.py가 중간에서 변환을 담당한다.

### bridge_ocr.py ROS2 토픽
| 토픽 | 방향 | 타입 | 설명 |
|------|------|------|------|
| `regist_car` | 구독 | String | AMR1 번호판 등록 요청 → `POST /api/vehicle/` |
| `request_car` | 구독 | String | AMR2 번호판 조회 요청 → `GET /api/vehicle/<id>/` |
| `servertoocr` | 발행 | String | 서버 조회 결과(plate_number) → OCR 노드로 전달 |
| `match_result` | 구독 | Bool | 일반 구역 매칭 결과 (False → 이벤트 삭제) |
| `match_result_id` | 구독 | String | 일반 구역 매칭 성공 + 이미지 → `WARNING_ISSUED` |
| `disabled` | 구독 | String | 장애인 차량 조회 요청 (번호판 텍스트) |
| `disabled_result` | 발행 | Bool | 장애인 차량 조회 결과 → OCR 노드로 전달 |
| `disabled_result_id` | 구독 | String | 장애인 구역 불법주차 → `WARNING_ISSUED` |
| `firecar_result_id` | 구독 | String | 소방차 구역 불법주차 → `WARNING_ISSUED` |
| `firecar_result` | 구독 | Bool | True=실제 소방차(정상주차) → 이벤트 삭제 / False=스킵(`firecar_result_id`에서 처리) |
| `/robot4/plate_id` | 구독 | String | AMR2 zone=4 수신 시 vehicle_id 저장 (`firecar_result` 처리 대비) |

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
