# Alley_Park_Patrol (차 빼!) 🚗🚨

> **AI 기반 골목길 및 소방 구역 불법 주정차 무인 순찰 및 단속 시스템**  
> AI-Powered Automated Illegal Parking Patrol & Enforcement System

### 프로젝트 개요
본 프로젝트는 고정식 웹캠 인프라, 중앙 관제 서버, 그리고 다중 자율주행 로봇(AMR)의 유기적인 협업을 통해 좁은 주택가 골목길과 소방차 전용 구역의 불법 주정차 문제를 해결하는 무인 관제 솔루션입니다.

---

## System Topology

### Webcam Infra
* **Hardware:** 고정형 웹캠
* **Software:** Ubuntu 22.04, OpenCV, YOLO, Python 3.10

### Central PC & Control Server
* **Hardware:** Server PC, Control PC (Ubuntu 22.04)
* **Software:** Python, Django Web Framework, Supabase  

### AMR 
* **Hardware:** TurtleBot4 플랫폼 (LiDAR, OAK-D Camera 탑재)
* **Software:** Ubuntu 22.04, ROS2 Humble, Nav2

---

### System Architecture & Scenario

본 시스템은 **3가지 관제 시나리오**를 기반으로 제작하였습니다.
1. **시나리오 A (일반 도로):** 골목길 황색 이중 실선 구역 시차 단속 
2. **시나리오 B (안전 구역):** 소방차 지정 구역 주정차 차량 단속
3. **시나리오 C (특수 구역):** 전용 주차 구역(장애인, 전기차 등) 미등록 차량 단속



---

### Daily Schedule 

| **Day** | **Task** | **Details** |
| :---: | :--- | :--- |
| **Day 1** | 시나리오 설계 | 팀 주제 확정, 3대 구체적 시나리오 및 요구사항(R1~R11) 정의 |
| **Day 2** | 세부 노드 및 아키텍처 | 레포지토리 생성, 시스템 토폴로지 및 파트별 R&R 확정, 시스템 설계도 작성 |
| **Day 3** | 파트별 프로토타이핑 | 웹캠 YOLO 추론 환경 구축 및 생성맵의 ROS2 슬램(SLAM) 테스트, 서버 구축 완료 |
| **Day 4** | 데이터 인터페이스 규격 정의 | 코드 리뷰, 비전 좌표 변환 데이터의 서버 전송 및 토픽 구조 및 ROS2 커스텀 메시지/액션 규격 정리 |
| **Day 5** | 데이터 파이프라인 통합 | PC-클라우드 DB 간 데이터 transaction 검증, 서버-로봇 간 제어 명령 연동 테스트 |
| **Day 6** | 전체 시스템 통합 및 최적화 | End-to-End 통합 테스트, 번호판 미인식 예외 처리 루틴 결합 및 최종 시스템 설계 문서(SDD) 완성 |
| **Day 7** | 자료 완성 및 리허설 | 시뮬레이션 구동 영상 촬영 및 편집, 발표 PPT 제작, 최종 시연 시나리오 데모 |
| **Day 8** | **최종 발표** | 프로젝트 최종 점검 및 오후 시연 발표 |

---

### Repository Structure
-# 상세 구현 및 구동 방법은 각 하위 폴더의 README를 참조.

```text
Alley_Park_Patrol/
├── webcam_infra/         # [Vision] 고정식 웹캠 비전 및 호모그래피 변환 패키지
├── central_server/       # [Server/DB] 관제 백엔드 서버 및 데이터베이스 
├── ros2_ws/              # [AMR] 로봇 주행 워크스페이스 (Nav2 튜닝, 순찰 및 단속 노드)
└── ai_models/            # [AI] YOLO 모델 학습 스크립트 및 가중치 관리 (LFS)
