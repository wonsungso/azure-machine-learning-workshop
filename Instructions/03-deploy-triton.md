# Lab 03 — Triton Inference Server로 모델 배포

---

## Lab 목표

이 Lab에서는 이전 단계에서 학습한 모델을 Azure Machine Learning의
Managed Online Endpoint로 배포하고 Triton Inference Server를 통해 추론을 수행합니다.

완료 후 상태:

- Managed Online Endpoint 생성
- Triton Deployment 구성
- 실시간 Inference 테스트

이 단계는 Workshop의 마지막 단계로, 학습된 모델을 실제 서비스 형태로 배포합니다.

---

## 사전 준비사항 (Before you start)

아래를 먼저 완료하세요:
- [00-setup.md](./00-setup.md) 완료
- [01-preprocess-data-rapids.md](./01-preprocess-data-rapids.md) 완료
- [02-train-model-pytorch.md](./02-train-model-pytorch.md) 완료
  - ✅ 모델 학습 완료
  - ✅ 모델 아티팩트 생성

> **참고**: 이 Lab에는 Compute Cluster가 불필요합니다. Managed Online Endpoint가 추론 환경을 제공합니다.

---

## Workshop 전체 흐름에서의 위치

```
Setup
   ↓
RAPIDS 데이터 전처리
   ↓
PyTorch 모델 학습
   ↓
[현재] Triton Endpoint 배포
```

---

## Triton Inference Server란?

NVIDIA Triton은 고성능 모델 추론을 위한 서버입니다.

특징:

- GPU 최적화 Inference
- ONNX / PyTorch / TensorRT 지원
- 실시간 API Endpoint 제공

간단히 말하면:

```
Training = 모델 생성
Triton   = 모델 서비스화
```

---

# 1️⃣ Managed Online Endpoint 생성

## Step 1. Endpoints 메뉴 이동

Azure ML Studio 좌측:

```
Assets → Endpoints
```

---

## Step 2. Online Endpoint 생성

```
+ Create → Real-time endpoint
```

설정:

```
Endpoint name : ep-dl-workshop
Authentication: Key
```

Create 클릭

---

## 생성 시간

약 2~3분

---

## ✅ Checkpoint

Endpoint 목록에 아래가 보이면 정상입니다.

```
ep-dl-workshop
```

---

# 2️⃣ Triton Deployment 생성

## Step 1. Deployment 추가

Endpoint 상세 화면:

```
+ Add deployment
```

설정:

```
Deployment name : triton-deploy
Instance type   : Standard_DS3_v2
Instance count  : 1
Model           : 학습된 모델 선택
```

Inference Server:

```
Triton
```

Create 클릭

---

## 💡 Workshop Tip

여기서 참가자에게 설명해 주세요:

```
Endpoint = API 주소
Deployment = 실제 실행되는 VM
```

---

## Deployment 준비 과정

내부적으로:

```
Container 생성
Model 로드
Triton 서버 시작
```

이 자동으로 수행됩니다.

---

## ✅ Checkpoint

Deployment 상태가 아래처럼 변경됩니다.

```
Creating → Healthy
```

Healthy 상태가 되면 준비 완료입니다.

---

# 3️⃣ Endpoint 테스트

## Step 1. Test 탭 이동

Endpoint 화면 상단:

```
Test
```

---

## Step 2. Sample Payload 입력

```json
{"input":[1,2,3]}
```

Run 클릭

---

## ✅ Checkpoint

Response JSON이 반환되면 성공입니다.

---

# 최종 아키텍처 구성

```
Azure ML Workspace
        ├── Compute Instance
        ├── GPU Compute Cluster
        └── Managed Online Endpoint
                └── Triton Deployment
```

---

# Workshop 진행 포인트

이 Lab에서 강조할 내용:

- Training과 Deployment는 완전히 다른 단계
- Azure ML은 모델을 바로 API로 배포 가능
- Triton은 GPU Inference 최적화 서버

---

# Workshop 종료 후 리소스 정리 (중요)

Azure Portal에서 Resource Group 삭제:

```
rg-aml-dl-workshop
```

GPU 및 Endpoint 비용을 방지할 수 있습니다.

---

---

# ⏹️ 실습 종료 후 리소스 정리

모든 실습을 완료한 후 **비용 절감**을 위해 리소스를 정리하세요.

## Step 1. Compute Instance 중지

Azure ML Studio:

```
Manage → Compute → Compute Instances → ci-aml-workshop → Stop
```

상태가 **Stopped**으로 변경되면 완료입니다.

## Step 2. Managed Online Endpoint 삭제 (선택사항)

Endpoint가 필요 없으면 삭제하여 비용 절감:

```
Assets → Endpoints → ep-dl-workshop → Delete
```

## Step 3. 전체 리소스 그룹 삭제 (선택사항)

Workshop을 완전히 정리할 경우 Azure Portal에서:

```
Resource Groups → rg-aml-dl-workshop → Delete resource group
```

이를 통해 모든 Azure 리소스(Workspace, Storage, Key Vault 등)가 제거됩니다.

---

# 🎉 Workshop 완료

축하합니다!

이번 Workshop에서 다음을 경험했습니다:

- Azure ML Workspace 구성
- RAPIDS GPU 데이터 전처리
- PyTorch GPU 학습
- Triton Endpoint 배포
