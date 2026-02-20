# 🚀 Azure Machine Learning Workshop - Basic

이 저장소는 Azure Machine Learning을 처음 접하는 사용자를 위한 **Hands-on Deep Learning Workshop** 입니다.

본 워크샵에서는 Azure ML Workspace 생성부터 GPU 학습, Triton Endpoint 배포까지 End-to-End 흐름을 실습합니다.

본 워크샵은 [Train compute-intensive models with Azure Machine Learning](https://github.com/MicrosoftLearning/mslearn-deep-learning) 를 기반으로 작성 되었습니다.

------------------------------------------------------------------------

# 🎯 Workshop 목표

이 Workshop을 완료하면 다음을 이해할 수 있습니다.

-   Azure Machine Learning Workspace 구조
-   RAPIDS 기반 GPU 데이터 전처리
-   PyTorch GPU Training Job 실행
-   Managed Online Endpoint 배포
-   Triton Inference Server 개념

------------------------------------------------------------------------

# 🧭 전체 실습 흐름

    Lab 00 — 환경 준비 (Setup)
            ↓
    Lab 01 — RAPIDS GPU 데이터 전처리
            ↓
    Lab 02 — PyTorch GPU 모델 학습
            ↓
    Lab 03 — Triton Endpoint 배포

------------------------------------------------------------------------

# 📂 Workshop 진행 순서

아래 순서대로 진행하세요.

## 1️⃣ 환경 준비

👉 Instructions/00-set-up.md

-   Resource Group 생성
-   Azure ML Workspace 생성
-   Compute Instance 생성

------------------------------------------------------------------------

## 2️⃣ RAPIDS 데이터 전처리

👉 Instructions/01-preprocess-data-rapids.md

-   Notebook 환경 준비
-   cuDF 기반 GPU 데이터 처리

------------------------------------------------------------------------

## 3️⃣ PyTorch 모델 학습

👉 Instructions/02-train-model-pytorch.md

-   GPU Compute Cluster 생성
-   Azure ML Job 실행
-   Training 로그 확인

------------------------------------------------------------------------

## 4️⃣ Triton Endpoint 배포

👉 Instructions/03-deploy-triton.md

-   Managed Online Endpoint 생성
-   Triton Deployment 구성
-   실시간 Inference 테스트

------------------------------------------------------------------------

# 🧱 Workshop 아키텍처 개요

    Azure ML Workspace
            ├── Compute Instance (Notebook)
            ├── GPU Compute Cluster (Training)
            └── Managed Online Endpoint (Inference)
                    └── Triton Server

------------------------------------------------------------------------

# ⚠️ 사전 준비사항

-   Azure Subscription
-   GPU VM quota (NC-series 권장)
-   Korea Central 또는 GPU 지원 리전

------------------------------------------------------------------------

# 🧹 Workshop 종료 후

반드시 Resource Group을 삭제하세요.

    rg-aml-dl-workshop

GPU 및 Endpoint 비용을 방지할 수 있습니다.

------------------------------------------------------------------------

# 👨‍💻 Workshop 스타일

본 Workshop은 다음을 목표로 설계되었습니다.

-   Instructor와 참석자가 동일한 문서 사용
-   Azure ML 입문자 기준 설명
-   발표 + 데모 Hybrid 진행 방식
-   Microsoft Learn(mslearn-deep-learning) 구조 기반

------------------------------------------------------------------------
