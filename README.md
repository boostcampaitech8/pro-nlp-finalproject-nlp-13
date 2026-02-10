# README

# 여행친구 (여친)

## 1. 프로젝트 개요

### 1.1 프로젝트 주제

 본 프로젝트는 딱딱한 비서형 LLM이 아니라, 카톡하듯 여행 계획을 완성해주는 친구형 여행 에이전트를 구현하였다. 단발성 체인이 아닌 State 를 유지하는 에이전트 구조를 이용하여 구축하는 것을 목표로 하였다.

### 1.2 주요 전략 및 방법론

 병렬적인 Tool동작, 정확성을 높이는 순환구조를 위해 LangGraph를 도입하였고, 에이전트의 친근한 답변을 위해 Qwen3-4B-Instruct 모델에 AI-Hub의 한국어 SNS 멀티턴 대화 데이터로  SFT와 DPO를 QLoRA 학습하여 챗봇Tool로 구현하였다

### 1.3 데이터셋

페르소나를 입히기 위한 대화 데이터셋과 RAG로 최신 여행 정보를 응답에 반영하기 위한 데이터셋 활용

- AI Hub 한국어 SNS 멀티턴 대화 데이터셋(여행/문화 카테고리) - 2인 대화, 총 38,334개 대화 추출[미정]
- 카카오맵 크롤링
    - 최신 정보 반영을 위해 2025년 기준 최신 리뷰 데이터 3개 크롤링
    - Selenium + BeautifulSoup 기반 동적 크롤링
- ‘비짓부산’ 홈페이지의 《블루리본 2025》, 《택슐랭 2025》 등 가이드북 16개
    - PyMuPDF 와 LLM을 이용해 데이터 정제 및 청크 구성

## **2. 팀원 소개 및 역할**

| 이름 | 프로필 | 역할 |
| :---: | :---: | --- |
| **김영현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/Kimyoung-hyun) | <img src="./assets/kim.jpg" width="100"> | RAG 구축, LangGraph 구현 |
| **윤준상**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/JunandSang) | <img src="./assets/yun.jpg" width="100"> | EDA, 프론트엔드 구현, Agent Tool 설계 |
| **장세현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/sucruba70) | <img src="./assets/jang.jpg" width="100"> | SFT 데이터 전처리,  LangGraph 구현, Agent Tool 설계 |
| **주현민**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/zoosumzoosum) | <img src="./assets/zoo.jpg" width="100"> | SFT데이터 수집/EDA , 페르소나 학습, 발표자료 제작 및 발표 |
| **한지석**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/jis-archive) | <img src="./assets/han.jpg" width="100"> | DPO 학습 데이터 구축, DPO 학습, RAG 평가 파이프라인 설계, 모델 서빙 |


## 3. 파이프라인

<img width="1000" alt="image" src="./assets/image.png">

<img width="1000" alt="image" src="./assets/image 2.png">

## 4. 디렉토리 구조

```bash
├─ server.py
│
src/
├─ AGENT/
│   ├─ __init__.py
│   ├─ graph.py
│   ├─ states.py
│   ├─ prompt
│   └─ tool
│
├─ RAG/
│   ├─ __init__.py
│   ├─ data_loader.py
│   ├─ hybrid_retriever.py
│   ├─ rag.py
│   ├─ rag_builder.py
│   └─ metric
│       └─ prompt.py
│
├─ common/
│   ├─ set_seed.py
│   └─ wandb.py
│
├─ dpo/
│   ├─ evaluate_chatbot.py
│   ├─ generate_dpo_dataset.py
│   ├─ train_dpo.py
│   ├─ data
│   │   ├─ custom_distilabel.py
│   │   └─ dpo_data_loader.py
│   └─ training
│       ├─ dpo_trainer.py
│       └─ model_loader.py
│
├─ sft/
│   ├─ train.py
│   ├─ data
│   │   ├─ collator.py
│   │   ├─ data_loader.py
│   │   └─ preprocessor.py
│   └─ training
│       ├─ model_loader.py
│       └─ trainer.py
│
frontend/
├─ package.json
├─ package-lock.json
├─ public/
│   ├─ favicon.ico
│   ├─ index.html
│   ├─ logo192.png
│   ├─ logo512.png
│   ├─ manifest.json
│   └─ robots.txt
│
├─ src/
│   ├─ App.css
│   ├─ App.js
│   ├─ App.test.js
│   ├─ background.jpg
│   ├─ bot_profile.png
│   ├─ index.css
│   ├─ index.js
│   ├─ logo.svg
│   ├─ reportWebVitals.js
│   └─ setupTests.js
│
├─ scripts/
│   ├─ generate_dataset
│   ├─ rag
│   ├─ serving
│   └─ train
│
configs/
│   ├─ dpo_config.yaml
│   ├─ rag.yaml
│   ├─ rag_metric.yaml
│   └─ sft_config.yaml
│
notebooks/
```

## 5. Train 및 Inference 실행

세팅을 직접하려면 `dpo_config.yaml`, `rag.yaml`, `rag_metric.yaml`, `sft_config.yaml` 를 참고해주세요.

**train**

```bash
# 파일 내 로직코드 실행
python -m scripts.{폴더명} {파일명} 

# 기본 사용법
python -m scripts.train --config {CONFIG_PATH}

# 모델 SFT 학습 예시
bash src/sft/train.py
# 모델 DPO 학습 예시
bash src/dpo/train_dpo.py
```

**inference**

```python
# 기본 사용법
python -m server.py

cd frontend
npm start

```

## 6. Wrap-Up Report

프로젝트 전반의 시행착오와 솔루션 및 회고는 [FinalProject_NLP_13.pdf](./assets/finalproject-nlp-13.pdf) 을 통해 확인할 수 있습니다.