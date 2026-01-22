# OvernightPredict

AI 기반 자율 코드 생성 시스템으로, 질문 예측과 자기 수정 정확도 메커니즘을 갖춘 엔터프라이즈급 프로젝트 빌더입니다.

## 핵심 기능

### 🤖 자율 밤샘 코딩
- 추가 지시 없이 자동으로 코딩 진행
- 프로젝트 컨텍스트 기반 자율 의사결정
- 컴포넌트 단위 점진적 구현

### 🔮 질문 예측 시스템
- **Context-Based**: 현재 컨텍스트 분석 기반 예측
- **Pattern Matching**: 과거 패턴 기반 예측
- **Semantic Similarity**: 의미적 유사도 기반 예측
- **Hybrid**: 다중 전략 결합

### 📊 정확도 평가 및 자기 수정
- 예측 질문과 실제 질문의 정확도 비교
- 정확도가 threshold 이하면 전략 자동 조정
- 정확도가 높으면 기존 예측대로 진행

### ⚡ 병렬 세션 실행
- 여러 세션 동시 실행으로 개발 속도 향상
- Auto-scaling 지원
- 세션 간 작업 분배

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Orchestrator                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Session 1  │  │  Session 2  │  │  Session N  │   ...   │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Overnight Engine                          │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Question │  Answer  │ Accuracy │ Strategy │     Code       │
│ Predictor│Generator │Evaluator │ Manager  │   Generator    │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/overnightPredict.git
cd overnightPredict

# 의존성 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

## 설정

1. `.env` 파일 생성:
```bash
cp .env.example .env
```

2. API 키 설정:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

## 사용법

### CLI 사용

```bash
# 밤샘 코딩 세션 시작
overnight start --name "MyProject" \
    --component auth \
    --component api \
    --component database \
    --sessions 3

# API 서버 시작
overnight serve

# 시스템 상태 확인
overnight status

# 새 프로젝트 초기화
overnight init
```

### Python API 사용

```python
import asyncio
from src.core.models import ProjectContext
from src.sessions.orchestrator import SessionOrchestrator
from src.core.config import get_settings

async def main():
    settings = get_settings()
    orchestrator = SessionOrchestrator(settings)

    # 프로젝트 정의
    project = ProjectContext(
        name="EnterpriseApp",
        description="Full-featured enterprise application",
        target_languages=["python", "typescript"],
        architecture_type="microservices",
        pending_components=[
            "authentication",
            "user_management",
            "api_gateway",
        ],
    )

    # 프로젝트 초기화 및 실행
    await orchestrator.initialize_project(project)
    await orchestrator.start(initial_sessions=3)

    # 모니터링...

    await orchestrator.stop()

asyncio.run(main())
```

### REST API

```bash
# 프로젝트 생성
curl -X POST http://localhost:8000/api/v1/projects \
    -H "Content-Type: application/json" \
    -d '{
        "name": "MyProject",
        "description": "Enterprise project",
        "components": ["auth", "api", "database"]
    }'

# 오케스트레이터 시작
curl -X POST http://localhost:8000/api/v1/orchestrator/start \
    -H "Content-Type: application/json" \
    -d '{"initial_sessions": 3}'

# 상태 확인
curl http://localhost:8000/api/v1/orchestrator/status

# 메트릭 조회
curl http://localhost:8000/api/v1/orchestrator/metrics
```

## 작동 원리

### 예측-검증 루프

```
1. 현재 컨텍스트 분석
        ↓
2. 다음 질문 예측 (5개)
        ↓
3. 예측된 질문에 대한 답변 사전 생성
        ↓
4. 실제 질문 도출/대기
        ↓
5. 예측 vs 실제 정확도 평가
        ↓
    ┌───┴───┐
    ↓       ↓
정확도 낮음   정확도 높음
    ↓           ↓
전략 변경    예측대로 진행
    ↓           ↓
새 예측 생성   코드 생성
    └───┬───┘
        ↓
6. 컨텍스트 업데이트
        ↓
    (반복)
```

### 전략 조정

정확도가 threshold(기본 70%) 이하로 떨어지면:
1. 현재 전략 성능 기록
2. 대안 전략 평가
3. 가장 적합한 전략으로 전환
4. 새로운 예측 생성

## 프로젝트 구조

```
overnightPredict/
├── src/
│   ├── core/
│   │   ├── engine.py       # 핵심 엔진
│   │   ├── models.py       # 데이터 모델
│   │   └── config.py       # 설정 관리
│   ├── predictors/
│   │   ├── question.py     # 질문 예측기
│   │   └── embeddings.py   # 임베딩 서비스
│   ├── generators/
│   │   ├── answer.py       # 답변 생성기
│   │   └── code.py         # 코드 생성기
│   ├── evaluators/
│   │   └── accuracy.py     # 정확도 평가기
│   ├── strategies/
│   │   └── manager.py      # 전략 관리자
│   ├── sessions/
│   │   ├── orchestrator.py # 세션 오케스트레이터
│   │   └── checkpoint.py   # 체크포인트 관리
│   ├── api/
│   │   ├── server.py       # FastAPI 서버
│   │   ├── routes.py       # API 라우트
│   │   └── websocket.py    # WebSocket 지원
│   └── cli.py              # CLI 인터페이스
├── tests/                  # 테스트
├── config/                 # 설정 파일
├── scripts/                # 실행 스크립트
└── data/                   # 데이터 저장소
```

## Docker 실행

```bash
# 이미지 빌드
docker build -t overnight-predict .

# 컨테이너 실행
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -p 8000:8000 \
    --env-file .env \
    overnight-predict

# Docker Compose 사용
docker-compose up -d
```

## 테스트

```bash
# 전체 테스트 실행
make test

# 커버리지 리포트 생성
make test-cov

# 린팅
make lint

# 포맷팅
make format
```

## 설정 옵션

`config/settings.yaml`:

```yaml
prediction:
  accuracy_threshold: 0.7      # 정확도 threshold
  lookahead_count: 5           # 미리 예측할 질문 수

sessions:
  max_parallel_sessions: 10    # 최대 병렬 세션
  auto_scale:
    enabled: true              # 자동 스케일링

ai:
  primary_provider: anthropic  # AI 제공자
```

## 라이선스

MIT License
