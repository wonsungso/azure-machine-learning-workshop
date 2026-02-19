# 🧪 Lab 03 — Triton Inference Server로 모델 배포

---

## 🎯 Lab 목표

이 Lab에서는 이전 단계에서 학습한 모델을 Azure Machine Learning의
Managed Online Endpoint로 배포하고 Triton Inference Server를 통해 추론을 수행합니다.

완료 후 상태:

- Managed Online Endpoint 생성
- Triton Deployment 구성
- 실시간 Inference 테스트

이 단계는 Workshop의 마지막 단계로, 학습된 모델을 실제 서비스 형태로 배포합니다.

---

## 🧭 Workshop 전체 흐름에서의 위치

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

## 🧠 Triton Inference Server란?

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

## ⏳ 생성 시간

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

## ⏳ Deployment 준비 과정

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

# 🧱 최종 아키텍처 구성

```
Azure ML Workspace
        ├── Compute Instance
        ├── GPU Compute Cluster
        └── Managed Online Endpoint
                └── Triton Deployment
```

---

# ⚠️ Troubleshooting

## ❌ Deployment 생성 실패

가능 원인:

- Instance quota 부족
- 모델 선택 오류

해결:

VM Size를 Standard_DS2_v2로 낮춰 재시도

---

## ❌ Endpoint 응답 없음

Deployment 상태가 Healthy인지 확인 후 재시도

---

# 🎤 Workshop 진행 포인트

이 Lab에서 강조할 내용:

- Training과 Deployment는 완전히 다른 단계
- Azure ML은 모델을 바로 API로 배포 가능
- Triton은 GPU Inference 최적화 서버

---

# 🧹 Workshop 종료 후 리소스 정리 (중요)

Azure Portal에서 Resource Group 삭제:

```
rg-aml-dl-workshop
```

GPU 및 Endpoint 비용을 방지할 수 있습니다.

---

# 🎉 Workshop 완료

축하합니다!

이번 Workshop에서 다음을 경험했습니다:

- Azure ML Workspace 구성
- RAPIDS GPU 데이터 전처리
- PyTorch GPU 학습
- Triton Endpoint 배포

작성일: 2026-02-19
