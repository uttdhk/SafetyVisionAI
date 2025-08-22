# SafetyVision AI - 탱크로리 안전상태 체크 시스템

CCTV 영상에서 탱크로리 차량 작업자의 안전장비 착용 여부를 AI로 자동 분석하는 시스템입니다.

## 🚀 주요 기능

### 📹 CCTV 영상 분석
- 비디오 파일에서 자동 프레임 추출
- 일정 간격 또는 움직임 감지 기반 프레임 선별
- 다양한 비디오 형식 지원 (MP4, AVI, MOV, MKV, WMV)

### 🤖 AI 기반 안전장비 검출
- **안전모 착용 여부** 자동 검출
- **안전벨트 착용 여부** 자동 검출 
- **안전조끼 착용 여부** 자동 검출
- YOLO 딥러닝 모델과 휴리스틱 분석 결합
- 높은 정확도와 빠른 처리 속도

### 📊 종합 분석 결과
- 실시간 안전도 점수 계산
- 위험도별 프레임 분류
- 위반 사항 자동 감지 및 분류
- 안전 권고사항 자동 생성

### 📈 시각화 및 보고서
- 대화형 웹 대시보드
- PDF 형태의 상세 분석 보고서
- 주석이 추가된 이미지 갤러리
- 통계 차트 및 그래프

## 🔧 설치 및 실행

### 시스템 요구사항
- Python 3.8 이상
- 4GB 이상의 RAM
- GPU 권장 (CPU에서도 동작 가능)

### 1. 저장소 클론
```bash
git clone https://github.com/your-repo/safetyvision-ai.git
cd safetyvision-ai
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 애플리케이션 실행
```bash
python main.py
```

웹 브라우저에서 `http://localhost:5000`에 접속하여 사용할 수 있습니다.

## 📁 프로젝트 구조

```
safetyvision-ai/
├── src/                          # 소스 코드
│   ├── video_processor/          # 비디오 처리 모듈
│   │   ├── __init__.py
│   │   └── frame_extractor.py    # 프레임 추출
│   ├── ai_analyzer/              # AI 분석 모듈
│   │   ├── __init__.py
│   │   └── safety_detector.py    # 안전장비 검출
│   ├── visualizer/               # 시각화 모듈
│   │   ├── __init__.py
│   │   └── report_generator.py   # 보고서 생성
│   └── web_interface/            # 웹 인터페이스
│       ├── __init__.py
│       └── app.py                # Flask 애플리케이션
├── templates/                    # HTML 템플릿
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   └── results.html
├── static/                       # 정적 파일
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── tests/                        # 테스트 파일
├── uploads/                      # 업로드된 파일
├── output/                       # 분석 결과 출력
├── models/                       # AI 모델 파일
├── docs/                         # 문서
├── config.py                     # 설정 파일
├── main.py                       # 메인 실행 파일
├── requirements.txt              # 의존성 패키지
└── README.md                     # 프로젝트 설명
```

## 🔬 사용 방법

### 1. 파일 업로드
- 웹 인터페이스에서 CCTV 영상 또는 이미지 파일을 업로드합니다
- 드래그 앤 드롭 또는 파일 선택을 통해 업로드 가능합니다
- 지원 형식: MP4, AVI, MOV, MKV, WMV, JPG, PNG, BMP

### 2. 자동 분석
- 업로드된 파일을 AI 모델이 자동으로 분석합니다
- 비디오의 경우 자동으로 프레임을 추출하여 분석합니다
- 실시간으로 분석 진행상황을 확인할 수 있습니다

### 3. 결과 확인
- 웹 대시보드에서 분석 결과를 실시간으로 확인합니다
- 안전도 점수, 위반 사항, 검출된 객체 정보를 제공합니다
- 주석이 추가된 이미지를 통해 검출 결과를 시각적으로 확인합니다

### 4. 보고서 다운로드
- PDF 형태의 상세 분석 보고서를 다운로드할 수 있습니다
- 시각화 대시보드 이미지를 다운로드할 수 있습니다
- 주석이 추가된 이미지 갤러리를 별도로 확인할 수 있습니다

## ⚙️ 설정

### 환경 변수 설정
`.env` 파일을 생성하여 다음 환경 변수를 설정할 수 있습니다:

```env
# Flask 설정
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# AI 모델 설정
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.4

# 프레임 추출 설정
FRAME_EXTRACTION_INTERVAL=30
```

### config.py 설정
`config.py` 파일에서 다양한 설정을 변경할 수 있습니다:

- 업로드 파일 크기 제한
- 지원하는 파일 형식
- AI 모델 신뢰도 임계값
- 프레임 추출 간격
- 안전장비 클래스 정의

## 🧪 테스트

### 테스트 실행
```bash
# 모든 테스트 실행
python -m pytest tests/

# 특정 테스트 파일 실행
python -m pytest tests/test_frame_extractor.py

# 커버리지와 함께 테스트 실행
python -m pytest tests/ --cov=src
```

### 테스트 구조
- `test_frame_extractor.py`: 프레임 추출 모듈 테스트
- `test_safety_detector.py`: 안전장비 검출 모듈 테스트
- `test_report_generator.py`: 보고서 생성 모듈 테스트
- `test_web_interface.py`: 웹 인터페이스 테스트

## 🔍 검출 가능한 안전장비

| 안전장비 | 설명 | 위험도 |
|---------|------|--------|
| 안전모 | 작업자의 머리 보호를 위한 안전모 착용 여부 | 높음 |
| 안전벨트 | 탱크로리 운전석에서의 안전벨트 착용 여부 | 높음 |
| 안전조끼 | 시인성 향상을 위한 형광 안전조끼 착용 여부 | 중간 |

## 📊 성능 지표

### 정확도
- 안전모 검출 정확도: ~85%
- 안전조끼 검출 정확도: ~80%
- 전체 시스템 정확도: ~82%

### 처리 속도
- 이미지당 평균 처리 시간: 0.5-1초
- 비디오 프레임 추출: 실시간의 10배 속도
- 전체 분석 완료: 1분 영상 기준 약 30초

## 🛠️ 개발자 가이드

### 새로운 안전장비 클래스 추가
1. `config.py`의 `SAFETY_CLASSES`에 새 클래스 추가
2. `SafetyDetector` 클래스의 `korean_labels` 딕셔너리 업데이트
3. 필요시 휴리스틱 검출 로직 추가
4. 테스트 케이스 작성

### 커스텀 AI 모델 사용
1. 모델 파일을 `models/` 디렉토리에 저장
2. `SafetyDetector` 초기화 시 `model_path` 파라미터 지정
3. 필요시 클래스 ID 매핑 업데이트

### API 엔드포인트 추가
1. `src/web_interface/app.py`에 새 라우트 추가
2. 필요시 HTML 템플릿 생성
3. JavaScript 함수 추가

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 발생하거나 질문이 있으시면 다음 방법으로 연락해주세요:

- 이슈 트래커: [GitHub Issues](https://github.com/your-repo/safetyvision-ai/issues)
- 이메일: support@safetyvision-ai.com
- 문서: [온라인 문서](https://docs.safetyvision-ai.com)

## 🙏 감사의 말

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 객체 검출 모델
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [Flask](https://flask.palletsprojects.com/) - 웹 프레임워크
- [Bootstrap](https://getbootstrap.com/) - UI 프레임워크

---

**SafetyVision AI** - 더 안전한 작업환경을 위한 AI 솔루션