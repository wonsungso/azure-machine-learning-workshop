# 🧪 Lab 00 — Azure Machine Learning 환경 준비 (Setup)

---

## 🎯 Lab 목표

이 Lab에서는 Deep Learning 실습을 위한 Azure Machine Learning 기본 환경을 구성합니다.

완료 후 상태:

- Resource Group 생성
- Azure ML Workspace 생성
- AML Studio 접속
- Compute Instance 생성

이 단계는 이후 RAPIDS 전처리 및 PyTorch GPU 학습을 위한 준비 단계입니다.

---

## 🧭 Workshop 전체 흐름에서의 위치

```
Setup
   ↓
RAPIDS 데이터 전처리
   ↓
PyTorch 모델 학습
   ↓
Triton Endpoint 배포
```

---

## 🧠 Azure Machine Learning 한 줄 개념

Azure ML Workspace는 ML 작업을 위한 관리 플랫폼입니다.

Workspace 생성 시 자동 생성 리소스:

- Storage Account
- Key Vault
- Container Registry
- Application Insights

---

# 1️⃣ Resource Group 생성

Azure Portal → Resource groups → Create

```
Resource group name : rg-aml-dl-workshop
Region              : Korea Central
```

---

# 2️⃣ Azure Machine Learning Workspace 생성

Azure Portal 검색:

```
Azure Machine Learning
```

설정:

```
Workspace name : aml-dl-workshop
Resource group : rg-aml-dl-workshop
Region         : Korea Central
```

생성 후 **Launch Studio** 클릭

---

# 3️⃣ Azure ML Studio 접속

Workspace 화면에서 Launch Studio 클릭 후 다음 메뉴 확인:

```
Author
Assets
Manage
```

---

# 4️⃣ Compute Instance 생성

좌측 메뉴:

```
Manage → Compute
```

생성:

```
Name    : ci-aml-workshop
VM Size : Standard_DS3_v2
```

상태가 Running 이 되면 완료입니다.

---

# 🧱 현재 아키텍처

```
Azure ML Workspace
        └── Compute Instance
```

---

# ⚠️ Troubleshooting

## Compute Instance 생성 실패

VM quota 부족 시:

```
Standard_DS2_v2 사용
```

---

# ▶️ Next Lab

```
01-preprocess-data-rapids.md
```

작성일: 2026-02-19
