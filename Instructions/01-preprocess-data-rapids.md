# 🧪 Lab 01 — RAPIDS를 활용한 GPU 데이터 전처리

---

## 🎯 Lab 목표

이 Lab에서는 RAPIDS를 활용하여 GPU 기반 데이터 전처리를 수행합니다.

완료 후 상태:

- Azure ML Notebook 환경 이해
- RAPIDS (cuDF) 기반 데이터 처리
- CPU vs GPU 데이터 처리 개념 이해

이 단계는 이후 PyTorch 모델 학습을 위한 데이터 준비 과정입니다.

---

## 🧭 Workshop 전체 흐름에서의 위치

```
Setup
   ↓
[현재] RAPIDS 데이터 전처리
   ↓
PyTorch 모델 학습
   ↓
Triton Endpoint 배포
```

---

## 🧠 RAPIDS란 무엇인가요?

RAPIDS는 NVIDIA에서 제공하는 GPU 데이터 사이언스 라이브러리입니다.

대표 구성:

- cuDF → GPU 기반 pandas
- cuML → GPU 기반 ML
- cuGraph → GPU 그래프 처리

간단히 말하면:

```
pandas = CPU 데이터 처리
cuDF   = GPU 데이터 처리
```

---

# 1️⃣ Notebook Repository 준비

## Step 1. Azure ML Studio 이동

왼쪽 메뉴:

```
Author → Notebooks
```

---

## Step 2. Terminal 열기

상단 메뉴에서 Terminal 실행 후 아래 명령어 입력:

```
git clone https://github.com/MicrosoftLearning/mslearn-deep-learning
```

---

## ✅ Checkpoint

Notebooks 목록에 아래 폴더가 생성되면 정상입니다.

```
mslearn-deep-learning
```

---

# 2️⃣ RAPIDS Notebook 실행

아래 경로의 Notebook을 엽니다.

```
01-preprocess-data-rapids.ipynb
```

Kernel 선택:

```
Python 3.10 (RAPIDS 환경)
```

---

## 💡 Workshop Tip

Kernel이 보이지 않는 경우:

- Compute Instance가 Running 상태인지 확인
- Kernel 재연결 (Reconnect)

---

# 3️⃣ 데이터 로드 및 GPU DataFrame 생성

Notebook을 위에서부터 순서대로 실행합니다.

핵심 코드 개념:

```python
import cudf
df = cudf.read_csv("data.csv")
```

설명:

- pandas 대신 cudf 사용
- 데이터가 GPU 메모리에 로드됩니다.

---

## 🧠 왜 GPU 전처리를 사용할까요?

Deep Learning에서는 데이터 크기가 매우 큽니다.

GPU를 활용하면:

- 데이터 로딩 속도 향상
- Feature Engineering 가속
- 학습 전처리 시간 단축

---

# 4️⃣ 데이터 전처리 결과 확인

Notebook 실행 후 다음을 확인합니다.

- DataFrame shape 출력
- GPU memory 사용 로그
- 변환된 Feature 컬럼 확인

---

## ✅ Checkpoint

아래와 같은 출력이 보이면 정상입니다.

```
GPU DataFrame created
Rows processed successfully
```

---

# 🧱 현재까지 구성된 아키텍처

```
Azure ML Workspace
        └── Compute Instance
                └── RAPIDS Notebook 실행
```

다음 Lab에서는 GPU Compute Cluster가 추가됩니다.

---

# ⚠️ Troubleshooting

## ❌ Notebook이 느리게 실행됨

가능 원인:

- Kernel이 CPU 환경으로 선택됨

해결:

```
RAPIDS Kernel로 변경 후 재실행
```

---

## ❌ cudf import 오류

Compute Instance 재시작 후 다시 실행

---

# 🎤 Workshop 진행 포인트

이 Lab에서 강조할 내용:

- RAPIDS는 pandas와 유사하지만 GPU 사용
- 데이터 전처리도 GPU에서 수행 가능
- Azure ML Notebook 환경은 별도 설치 없이 사용 가능

---

# ▶️ Next Lab

```
02-train-model-pytorch.md
```

GPU Compute Cluster를 사용하여 PyTorch 모델 학습을 진행합니다.

작성일: 2026-02-19
