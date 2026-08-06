# Day 1 — 제조 시계열 AI

「제조산업에서 적용하는 AI 기술」 3일 과정 중 **첫째 날** 실습 자료입니다.
Transformer부터 표현학습·도메인 적응까지, 제조 센서 시계열을 다루는 흐름을 다섯 권의 노트북으로 따라갑니다.

---

## 실습 시작하기

아래 링크를 누르면 Google Colab에서 바로 열립니다.
**여는 즉시 `파일 → 드라이브에 사본 저장`을 눌러 주세요.** 그래야 작성한 코드가 남습니다.

| # | 노트북 | 주제 | Colab |
|---|---|---|---|
| 00 | `d1_00_manufacturing_ai_landscape` | 제조 AI 비즈니스 지형과 3대 난제 | [열기](https://colab.research.google.com/github/leejiyoon52/ai-course/blob/main/day1/notebooks/d1_00_manufacturing_ai_landscape.ipynb) |
| 01 | `d1_01_timeseries_foundations` | 시계열 데이터의 이해와 실습 토대 | [열기](https://colab.research.google.com/github/leejiyoon52/ai-course/blob/main/day1/notebooks/d1_01_timeseries_foundations.ipynb) |
| 02 | `d1_02_deep_timeseries_evolution` | 딥러닝 시계열 계보 — RNN에서 PatchTST까지 | [열기](https://colab.research.google.com/github/leejiyoon52/ai-course/blob/main/day1/notebooks/d1_02_deep_timeseries_evolution.ipynb) |
| 03 | `d1_03_anomaly_transformer_xai` | Anomaly Transformer와 원인 추적 (XAI) | [열기](https://colab.research.google.com/github/leejiyoon52/ai-course/blob/main/day1/notebooks/d1_03_anomaly_transformer_xai.ipynb) |
| 04 | `d1_04_ssl_domain_adaptation` | 표현학습(SSL)과 도메인 적응 | [열기](https://colab.research.google.com/github/leejiyoon52/ai-course/blob/main/day1/notebooks/d1_04_ssl_domain_adaptation.ipynb) |

### 시작 전 확인

1. **런타임 → 런타임 유형 변경 → T4 GPU** 로 바꿉니다. (나중에 바꾸면 세션이 초기화됩니다)
2. 위에서부터 셀을 실행합니다. 데이터와 모듈은 이 저장소에서 자동으로 받아집니다.
3. **API 키는 필요 없습니다.**

---

## 폴더 구조

```
day1/
├── notebooks/     실습 노트북 5권
├── modules/       mfg_datagen.py (합성 데이터 생성기) · loaders.py (데이터 로더)
├── cmapss/        NASA C-MAPSS 터보팬 엔진 데이터 (FD001, FD003)
├── pump/          수처리장 펌프 센서 5분 요약본
├── anomaly/       펌프 테스트베드 이상탐지 데이터
└── checkpoints/   사전학습 모델 가중치 (히트맵·t-SNE 품질용)
```

---

## 데이터 출처

실습에는 **공개 실데이터 3종**과 **합성 데이터**가 함께 쓰입니다.

| 폴더 | 원본 | 출처 |
|---|---|---|
| `cmapss/` | NASA Turbofan Engine Degradation Simulation (C-MAPSS) | NASA Prognostics Center of Excellence 공개 데이터 |
| `pump/` | Pump Sensor Data (수처리장 펌프 52센서·5개월) | Kaggle 공개 데이터셋을 5분 평균·12센서로 축약 |
| `anomaly/` | SKAB — Skoltech Anomaly Benchmark (수순환 펌프 테스트베드) | https://github.com/waico/SKAB |

합성 데이터(사출성형기·CNC 스핀들·라인 전력·압출기)는 `modules/mfg_datagen.py` 가 생성합니다.
전처리 함정(결측 블록·홀드값)과 이상 3유형, 그리고 **원인 센서 정답**을 의도적으로 심어야 하는
실습에서는 합성 데이터가 필요합니다.

> 원본 데이터의 저작권은 각 제공처에 있습니다. 이 저장소의 파일은 **교육 목적의 축약본**이며,
> 상업적 재배포 전에는 각 출처의 이용 조건을 확인하시기 바랍니다.

---

## 네트워크가 끊겨도 진행됩니다

강의장에서 다운로드가 실패하면 `loaders.py` 가 **같은 스키마의 합성 데이터를 즉석에서 생성**해
노트북이 끝까지 실행됩니다. 이때 이런 메시지가 출력됩니다.

```
⚠️ 실데이터 수신 실패 — 합성 데이터로 진행합니다
```

컬럼·자료형·형태는 실데이터와 동일하지만 **숫자는 달라집니다.**
슬라이드에 적힌 수치가 아니라 **화면에 출력된 숫자**를 읽어 주세요.

---

## 3일 과정 안내

- **Day 1 — 제조 시계열 AI** ← 지금 여기
- Day 2 — LLM & Tabular: 정형 데이터 예측 / LLM Agent·RAG
- Day 3 — Vision & MLOps: 비전 검사 / 모델 경량화 / VFM
