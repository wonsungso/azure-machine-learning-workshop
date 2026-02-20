# Lab 02 — PyTorch 모델 학습 (GPU Compute Cluster)

---

## Lab 목표

이 Lab에서는 Azure Machine Learning의 GPU Compute Cluster를 사용하여
PyTorch 모델 학습을 수행합니다.

완료 후 상태:

- 기존 GPU Compute Cluster 재사용
- Azure ML Job 개념 이해
- PyTorch Training Job 실행
- 모델 아티팩트 생성 확인

이 단계는 Triton 배포 이전의 핵심 학습 단계입니다.

---

## 사전 준비사항 (Before you start)

아래를 먼저 완료하세요:
- [00-setup.md](./00-setup.md) 완료
- [01-preprocess-data-rapids.md](./01-preprocess-data-rapids.md) 완료
  - ✅ RAPIDS Environment 생성
  - ✅ 데이터 전처리 완료 (`processed_data.csv` 생성)

---

## Workshop 전체 흐름에서의 위치

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

## Azure ML Job 개념 (중요)

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

# 1️⃣ Compute 상태 확인

Lab 00에서 이미 생성한 GPU Compute Cluster를 그대로 사용합니다.

Azure ML Studio 좌측:

```
Manage → Compute
```

확인 항목:

```
Compute Cluster: cc-aml-gpu
```

> 클러스터가 Idle/Stopped처럼 보여도 정상입니다. Job 제출 시 자동으로 올라옵니다.

---

# 2️⃣ 학습 노트북 열기

Azure ML Studio 좌측:

```
Authoring → Notebooks
```

아래 경로의 노트북을 엽니다.

```
azure-machine-learning-workshop/Notebooks/02-train-model/02-train-model.ipynb
```

---

# 3️⃣ 노트북 셀 실행으로 Training Job 제출

위에서부터 셀을 순서대로 실행합니다.

체크 포인트:

- Workspace 로드 셀 실행 성공
- Environment 로드 셀 실행 성공
- `ScriptRunConfig` 셀에서 `compute_target`이 `cc-aml-gpu`인지 확인
- 제출 셀 실행 (`run.wait_for_completion(show_output=True)`)

> ⏳ Training Job은 약 20 분 소요 됩니다.

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

# 현재까지 구성된 아키텍처

```
Azure ML Workspace
        ├── Compute Instance
        └── GPU Compute Cluster
                └── Training Job 실행
```

---

# Workshop 진행 포인트

이 Lab에서 강조할 내용:

- Azure ML은 Kubernetes 기반 Job 실행 구조
- GPU는 필요할 때만 생성
- Notebook과 Job은 역할이 다름

---

---

# ⏹️ 실습 종료 후 리소스 중지

Compute Instance를 중지하여 불필요한 비용 발생을 방지하세요.

**Step 1. Compute 페이지 이동**

Azure ML Studio 좌측:

```
Manage → Compute → Compute Instances
```

**Step 2. Compute Instance 중지**

```
ci-aml-workshop → Stop
```

상태가 **Stopped**으로 변경되면 비용 청구 중단됩니다.

> 💡 **Compute Cluster는 이미 자동 종료됨** (idle 상태 시 자동 scale-down)

---

# ▶️ Next Lab

[03-deploy-triton.md](./03-deploy-triton.md)

학습된 모델을 Triton Inference Server로 배포합니다.
