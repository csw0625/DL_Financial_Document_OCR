# DL_Financial_Document_OCR
### SNU KDT 6조
### 권서영 이소민 이유림 조성우 지희선 홍예나

> **목표**: 금융권 신청/신고서 서식 기반 문서에서 **손글씨 영역을 탐지**하고 **문자를 인식**하여 디지털 텍스트로 변환  
> **프로세스**: Object Detection → Text Recognition → Prediction_Pipeline  
> **데이터**: [AI Hub의 OCR 데이터(금융 및 물류)](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM015&aihubDataSe=data&dataSetSn=71301)에서 6가지 금융 문서(신고서1개 신청서 5개) 양식 사용

<pre>
.
├── README.md
├── 발표자료.pdf
├── notebooks/
    ├── Baseline_CNN(객체탐지).ipynb
    ├── R-CNN.ipynb
    ├── CRAFT.ipynb
    ├── Baseline_CNN(문자인식).ipynb
    ├── C-RNN.ipynb
    ├── ViT.ipynb
    ├── Swin_Transformer.ipynb
    └── R_CNN+C_RNN_prediction.ipynb  ## Final prediction pipeline
</pre>
---

## 1. 프로젝트 개요
- **프로젝트명**: 금융 손글씨 OCR 파이프라인
- **프로젝트 소개**
  - 문서 이미지에서 손글씨만 **정확히 분리·탐지**하고, 인식 모델로 **문자열**을 생성
  - 금융 도메인의 높은 정확도 요구에 맞춰 **탐지 F1**과 **인식 CER**를 핵심 지표로 설정
  - 최종 **엔드-투-엔드 성능**을 위해 탐지-인식 조합을 비교하여 최적 조합을 도출
---

## 2. 분석

### 2.1 모델 평가 방법
- **탐지 지표**: `F1@IoU=0.80`  
  - IoU(Intersection over Union) ≥ 0.80에서 정·오탐을 계산하여 Precision/Recall, F1 산출
- **인식 지표**: `CER` (Character Error Rate)  
  - (치환 S + 삭제 D + 삽입 I) / 총문자수 N
- **엔드-투-엔드**: Detection 모델로 잘라낸 박스를 Recognition 모델에 투입 → 최종 문자열 CER

### 2.2 분석 Flow
- **Preprocessing**
  - 문서 이미지 정규화, 크롭/스케일(예: 1024×1024), 아티팩트 제거
  - 샘플 분포 점검, 학습/검증 분리(문서 양식/필기자 분리 유지)
<img width="1335" height="311" alt="output" src="https://github.com/user-attachments/assets/acceda34-d60c-4135-b112-0bf7e52c6977" />

- **Object Detection (객체탐지)**
  - **Baseline CNN(객체탐지)**: 직접 구현한 경량 CNN 기반 기본선
  - **R-CNN 계열**: 영역 제안 기반 2-Stage(ResNet-50 백본, RPN+ROI Heads)
  - **CRAFT**: 글자 영역에 특화된 Region/Affinity score 기반 텍스트 탐지
- **Text Recognition (문자인식)**
  - **C-RNN**: CNN feature → Bi-LSTM 시퀀스 인코딩 → CTC Loss
  - **ViT**: 패치 기반 Self-Attention, CTC 학습
  - **Swin Transformer**: Shifted window 기반 계층적 인코딩(초기 CTC, 버전별 CE 적용)
- **조합 평가**
  - 탐지 결과 박스 → 인식 입력 → CER 계산  
  - 조합별 CER 비교로 최종 파이프라인 결정

### 2.3 결과 요약
- **탐지 단일 성능**
  - *Baseline CNN(Detection)* F1: 0.1967  
  - **R-CNN F1: 0.999**  
  - *CRAFT* F1: 0.9783
- **인식 단일 성능 (CER, ↓가 좋음)**
  - *Baseline CNN(Recognition)*: 0.27  
  - **C-RNN: 0.021**  
  - *ViT*: 0.03  
  - *Swin*: 0.30
- **엔드-투-엔드 조합 성능 (CER)**
  - CRAFT + Swin: 0.513  
  - R-CNN + Swin: 0.824  
  - CRAFT + ViT: 0.270  
  - R-CNN + ViT: 0.064  
  - **R-CNN + C-RNN: 0.0207 ← 최종 선정**
<p align="center">
  <img src="https://github.com/user-attachments/assets/16e4d167-7d8f-4114-93f7-e11a2be99058"
       alt="손글씨_250910_162848-3_vis"
       width="600" height="800">
</p>
---

## 3. 결론
- **최종 파이프라인**: **R-CNN(탐지) + C-RNN(인식)**  
  - 금융 서식의 손글씨를 **정밀 탐지**하고 **낮은 CER**로 인식
- **유용성**
  - 양식 독립적으로 다양한 문서에서 **손글씨만 분리**하여 인식
  - 금융 현업의 수기 입력 처리 자동화에 적합
- **한계/개선**
  - 데이터 확장(필기체 다양성, 난이도 샘플)과 문맥 후처리(사전/룰, LLM 기반 오류 교정)
  - End-to-End 학습(탐지-인식 공동 최적화), ViT/Swin 안정화(스케줄러/정규화/사전학습 가중치)
