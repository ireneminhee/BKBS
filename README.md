# BKBS (Be Kinder Be Smarter) - 뉴스 추천 시스템

똑똑해지고 싶은 모두를 위한 친절한 뉴스 플랫폼입니다.

## 🚀 주요 기능

- **기사 요약 서비스**: AI 기반 기사 요약
- **주요 단어 의미 제공**: 어려운 용어 설명
- **단어 검색 서비스**: 실시간 단어 검색
- **AI 기반 기사 추천**: 요약 기반 코사인 유사도 추천
- **대화 기록 기반 AI 추천**: 사용자 대화를 분석한 개인화 추천
- **생성 AI와 토론 서비스**: 기사에 대한 AI와의 토론
- **AI 이미지 생성**: 기사 요약 기반 이미지 생성

## 📋 설치 및 설정

### 1. 가상환경 활성화
```bash
source myenv/bin/activate
```

### 2. OpenAI API 키 설정

#### 방법 1: 설정 스크립트 사용 (권장)
```bash
./setup_api_key.sh
```
스크립트를 실행하고 API 키를 입력하면 `.env` 파일에 안전하게 저장됩니다.

#### 방법 2: 수동으로 .env 파일 생성
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

#### 방법 3: 환경 변수 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 방법 4: OpenAI 웹사이트에서 API 키 생성
1. [OpenAI Platform](https://platform.openai.com/account/api-keys) 접속
2. 새 API 키 생성
3. 위 방법 중 하나로 설정

### 3. 애플리케이션 실행
```bash
python main.py
```

## 🔧 추천 시스템 비교

### 코사인 유사도 기반 추천
- **알고리즘**: Word2Vec + 코사인 유사도
- **데이터**: 기사 요약
- **특징**: 객관적, 빠른 응답, 수학적 검증 가능

### 대화 기록 기반 AI 추천
- **알고리즘**: GPT-3.5-turbo 자연어 처리
- **데이터**: 사용자 대화 기록
- **특징**: 개인화, 맥락적 이해, 추천 이유 설명

## 📊 사용 방법

1. **웹 애플리케이션 실행**
   - `python main.py` 실행
   - 브라우저에서 `http://localhost:5000` 접속

2. **기사 탐색**
   - 기사 목록에서 원하는 기사 클릭
   - 요약, 주요 단어, 관련 기사 확인

3. **토론 챗봇 사용**
   - 기사 페이지 하단의 토론 챗봇 클릭
   - AI와 기사에 대해 대화

4. **추천 시스템 비교**
   - 대화 진행 후 추천 버튼 클릭
   - 두 추천 시스템 결과 비교

## 🛠️ 기술 스택

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **AI/ML**: OpenAI GPT-3.5-turbo, Word2Vec, scikit-learn
- **Chatbot**: Streamlit
- **Data Processing**: pandas, numpy, konlpy

## 📁 프로젝트 구조

```
BKBS/
├── .gitignore              # Git 제외 파일 목록
├── .env                    # API 키 (Git에 업로드되지 않음)
├── README.md               # 프로젝트 문서
├── requirements.txt        # Python 패키지 의존성
├── setup_api_key.sh        # API 키 설정 스크립트
├── env_example.txt         # 환경 변수 예시
├── main.py                 # Flask 메인 애플리케이션
├── streamlit_chatbot.py   # 토론 챗봇
├── data/
│   └── data.csv           # 기사 데이터
├── static/                # 정적 파일
│   ├── styles.css
│   ├── scripts.js
│   └── img/
├── templates/             # HTML 템플릿
│   ├── index.html
│   ├── article.html
│   └── article_list.html
└── utils/                  # 유틸리티
    ├── chatgpt_api.py
    └── word2vec_model.model

```

## ⚠️ 주의사항

- OpenAI API 키가 필요합니다
- API 사용량에 따라 비용이 발생할 수 있습니다
- Word2Vec 모델은 한국어 뉴스 데이터로 학습되었습니다

## 🤝 기여

이 프로젝트는 교육 목적으로 제작되었습니다. 개선 사항이나 버그 리포트는 언제든 환영합니다.

## 📄 라이선스

이 프로젝트는 교육 목적으로 자유롭게 사용할 수 있습니다.
