# 🧪 Lab 02 — PyTorch 모델 학습 (GPU Compute Cluster)

---

## 🎯 Lab 목표

이 Lab에서는 Azure Machine Learning의 GPU Compute Cluster를 사용하여
PyTorch 모델 학습을 수행합니다.

완료 후 상태:

- GPU Compute Cluster 생성
- Azure ML Job 개념 이해
- PyTorch Training Job 실행
- 모델 아티팩트 생성 확인

이 단계는 Triton 배포 이전의 핵심 학습 단계입니다.

---

## 🧭 Workshop 전체 흐름에서의 위치

```
Setup
   ↓
RAPIDS 데이터 전처리
   ↓
[현재] PyTorch 모델 학습
   ↓
Triton Endpoint 배포
```

---

## 🧠 Azure ML Job 개념 (중요)

Azure ML에서 모델 학습은 **Job**이라는 단위로 실행됩니다.

Job 구성 요소:

- Code (학습 코드)
- Environment (라이브러리)
- Compute (GPU Cluster)
- Inputs/Outputs (데이터)

즉,

```
Job = 학습 실행 요청
```

이라고 이해하시면 됩니다.

---

# 1️⃣ GPU Compute Cluster 생성

## Step 1. Compute 메뉴 이동

Azure ML Studio 좌측:

```
Manage → Compute
```

---

## Step 2. Compute Cluster 생성

```
+ New → Compute Cluster
```

설정값:

```
Name       : gpu-cluster
VM Size    : Standard_NC6s_v3
Min Nodes  : 0
Max Nodes  : 2
```

Create 클릭

---

## 💡 Workshop Tip

GPU VM은 항상 켜져 있지 않습니다.

```
Job 실행 시 자동 생성
Job 종료 시 자동 종료
```

비용 최적화를 위한 구조입니다.

---

## ⏳ 생성 시간

약 2~5분

---

## ✅ Checkpoint

Compute 목록에 아래가 보이면 정상입니다.

```
gpu-cluster — Idle
```

---

# 2️⃣ Training 코드 확인

Notebook 또는 Repository에서 아래 파일을 확인합니다.

```
train.py
```

이 파일은 PyTorch 학습을 수행하는 스크립트입니다.

---

## 🧠 왜 Notebook이 아니라 Job으로 학습하나요?

Notebook은 개발 환경이고,

```
실제 학습 = Job
```

으로 실행하는 것이 Azure ML의 권장 방식입니다.

---

# 3️⃣ Training Job 생성

## Step 1. Jobs 메뉴 이동

좌측 메뉴:

```
Author → Jobs
```

---

## Step 2. Command Job 생성

```
+ Create → Command Job
```

설정:

```
Compute : gpu-cluster
Command : python train.py
```

코드 경로는 mslearn-deep-learning repo 위치를 선택합니다.

Submit 클릭

---

## ⏳ Job 실행 흐름

Job 실행 시 내부적으로:

```
GPU VM 생성
Docker 환경 준비
학습 실행
로그 수집
```

이 자동으로 진행됩니다.

---

## ✅ Checkpoint

Job 상태가 아래 순서로 변경됩니다.

```
Queued → Preparing → Running → Completed
```

Running 상태가 보이면 GPU가 생성된 것입니다.

---

# 4️⃣ 학습 로그 확인

Job 상세 화면에서:

```
Outputs + logs
```

클릭

확인 내용:

- Epoch 로그
- Loss 출력
- GPU 사용 로그

---

## 💡 Workshop Tip

참석자에게 꼭 보여주세요:

```
GPU가 자동 생성되는 모습
```

Azure ML의 핵심 경험입니다.

---

# 🧱 현재까지 구성된 아키텍처

```
Azure ML Workspace
        ├── Compute Instance
        └── GPU Compute Cluster
                └── Training Job 실행
```

---

# ⚠️ Troubleshooting

## ❌ GPU VM 생성 실패

가능 원인:

- GPU quota 부족

해결:

Azure Portal → Quota 증가 요청

---

## ❌ Job이 Preparing에서 멈춤

보통 Docker 이미지 pull 중입니다.
몇 분 기다린 후 새로고침

---

# 🎤 Workshop 진행 포인트

이 Lab에서 강조할 내용:

- Azure ML은 Kubernetes 기반 Job 실행 구조
- GPU는 필요할 때만 생성
- Notebook과 Job은 역할이 다름

---

# ▶️ Next Lab

```
03-deploy-triton.md
```

학습된 모델을 Triton Inference Server로 배포합니다.

작성일: 2026-02-19
