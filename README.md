# Alley_Park_Patrol (차 빼!) 🚗🚨

> **AI 기반 골목길 및 소방 구역 불법 주정차 무인 순찰 및 단속 시스템**  
> AI-Powered Automated Illegal Parking Patrol & Enforcement System

### 프로젝트 개요
본 프로젝트는 고정식 웹캠 인프라, 중앙 관제 서버, 그리고 다중 자율주행 로봇(AMR)의 유기적인 협업을 통해 좁은 주택가 골목길과 소방차 전용 구역의 불법 주정차 문제를 해결하는 스마트시티 무인 관제 솔루션입니다.

---

## System Topology

### Webcam Infra
* **Hardware:** 고정형 웹캠 (Logitech C920급) / IP CCTV 카메라
* **Software:** Ubuntu 22.04, OpenCV, YOLO, Python 3.10

### Central PC & Control Server
* **Hardware:** 중앙 관제 마스터 PC (Ubuntu 22.04)
* **Software:** Python (FastAPI/Flask), MySQL 

### AMR 
* **Hardware:** TurtleBot4 플랫폼 (LiDAR, OAK-D Camera 탑재)
* **Software:** Ubuntu 22.04, ROS2 Humble, Nav2, Dynamic Reconfigure

---

### 시스템 아키텍처 & 시나리오

본 시스템은 **3가지 구체적 관제 시나리오**를 기반으로 구동됩니다.
1. **시나리오 A (일반 도로):** 골목길 황색 이중 실선 구역 시차 단속 
2. **시나리오 B (안전 구역):** 소방차 지정 구역 즉시 대피 및 사이렌/방송을 통한 강제 통제
3. **시나리오 C (특수 구역):** 전용 주차자리(장애인, 전기차 등) 주차선 색상 및 클래스 매칭 단속



---

### Daily Schedule 

| **Day** | **Task** | **Details** |
| :---: | :--- | :--- |
| **Day 1** | 주제 확정 및 시나리오 설계 | 팀명 '차 빼!' 확정, 3대 구체적 관제 시나리오 및 요구사항(R1~R11) 정의 |
| **Day 2** | **저장소 구축 및 아키텍처 수립** | `Alley_Park_Patrol` 레포지토리 생성, 시스템 토폴로지 및 파트별 R&R 확정 |
| **Day 3** | 파트별 프로토타이핑 | 웹캠 YOLO 추론 환경 구축 및 ROS2 주택가 가상 환경(Gazebo) 맵 슬램(SLAM) 테스트 |
| **Day 4** | 데이터 인터페이스 정의 | 서버-로봇 간 통신을 위한 MQTT 토픽 구조 및 ROS2 커스텀 메시지/액션 규격 설정 |
| **Day 5** | 파이프라인 통합 및 분석 | 비전 좌표 변환 데이터의 서버 전송 및 로봇 주행 제어 명령 연동 테스트 |
| **Day 6** | 발표 자료 작성 및 백업 | 시나리오 구현 프로토타입 시뮬레이션 영상 확보 및 발표 장표(PPT) 제작 |
| **Day 7** | 데모 리허설 | 좁은 골목길 로봇 간 충돌 데드락 방지 등 예외 상황 시나리오 최종 점검 |
| **Day 8** | **최종 발표** | 프로젝트 시스템 설계 아키텍처 및 핵심 통신 파이프라인 구축 결과 발표 |

---

### 저장소 구조 (Repository Structure)
*상세 구현 및 구동 방법은 각 하위 폴더의 README를 참조.*

```text
Alley_Park_Patrol/
├── webcam_infra/         # [Vision] 고정식 웹캠 비전 및 호모그래피 변환 패키지
├── central_server/       # [Server/DB] 관제 백엔드 서버 및 데이터베이스 
├── ros2_ws/              # [AMR] 로봇 주행 워크스페이스 (Nav2 튜닝, 순찰 및 단속 노드)
└── ai_models/            # [AI] YOLO 모델 학습 스크립트 및 가중치 관리 (LFS)
