from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
import ast
import subprocess
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# from konlpy.tag import Okt  # 사용하지 않음
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Streamlit 앱 경로
STREAMLIT_APP_PATH = "streamlit_chatbot.py"

# OpenAI API 키 설정 (환경 변수에서 가져오거나 직접 설정)
import os
import json
import pickle
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# 캐시 파일 경로
CACHE_FILE = "summary_cache.pkl"

# def load_summary_cache():
#     """캐시 파일에서 요약 캐시를 불러옵니다."""
#     try:
#         if os.path.exists(CACHE_FILE):
#             with open(CACHE_FILE, 'rb') as f:
#                 cache = pickle.load(f)
#                 print(f"캐시 로드 완료: {len(cache)}개 요약")
#                 return cache
#     except Exception as e:
#         print(f"캐시 로드 실패: {e}")
#     return {}

# def save_summary_cache(cache):
#     """요약 캐시를 파일에 저장합니다."""
#     try:
#         with open(CACHE_FILE, 'wb') as f:
#             pickle.dump(cache, f)
#             print(f"캐시 저장 완료: {len(cache)}개 요약")
#     except Exception as e:
#         print(f"캐시 저장 실패: {e}")

# Streamlit 앱 실행
def run_streamlit():
    # Streamlit 앱을 백그라운드에서 실행
    subprocess.Popen(
        ["streamlit", "run", STREAMLIT_APP_PATH],
        stdout=subprocess.DEVNULL,  # 출력 숨기기
        stderr=subprocess.DEVNULL,
        shell=False,
    )

# CSV 파일에서 데이터 읽기
def load_articles():
    df = pd.read_csv('data/news_data_2.csv')  # 데이터 파일 읽기
    # processed_text 컬럼 동적 생성 (title + content)
    articles = []

    # iterrows()는 index와 row를 반환합니다.
    for index, row in df.iterrows():
        article = {
            'id': index,  # 'id'를 index로 설정
            'date': row['date'],
            'journalist': row['journalist'],
            'source': row['source'],
            'title': row['title'],
            'content': row['content']
        }
        articles.append(article)

    return articles


# `ast.literal_eval`을 안전하게 처리하는 함수. 문자열을 파이썬 객체로 반환하는 역할
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def get_text_embeddings(articles_data, sentence_model):
    """
    SentenceTransformer를 사용하여 텍스트 임베딩 생성
    """
    # 모든 기사의 텍스트를 리스트로 추출
    texts = []
    for article in articles_data:
        # title + content 결합
        combined_text = article['title'] + " " + article['content']
        texts.append(combined_text)
    
    # SentenceTransformer로 임베딩 생성
    embeddings = sentence_model.encode(texts, convert_to_tensor=False)
    
    return embeddings



def get_text_based_recommendations(clicked_article_id, articles_data, sentence_model):
    """
    SentenceTransformer 기반 텍스트 추천 시스템
    """
    # 클릭한 기사 가져오기
    clicked_article = next((a for a in articles_data if a['id'] == clicked_article_id), None)

    if clicked_article is None:
        return []  # 클릭한 기사를 찾을 수 없으면 빈 리스트 반환

    # 클릭한 기사의 텍스트 임베딩 계산
    clicked_text = clicked_article['title'] + " " + clicked_article['content']
    clicked_embedding = sentence_model.encode([clicked_text], convert_to_tensor=False)

    # 모든 기사의 텍스트 임베딩 계산
    text_embeddings = get_text_embeddings(articles_data, sentence_model)

    # 클릭한 기사와 다른 기사들의 코사인 유사도 계산
    similarities = cosine_similarity(clicked_embedding, text_embeddings).flatten()

    # 유사도가 높은 순으로 기사 추천 (자기 자신은 제외)
    recommended_articles = sorted(
        [(i, sim) for i, sim in enumerate(similarities) if articles_data[i]['id'] != clicked_article_id],
        key=lambda x: x[1],
        reverse=True
    )

    # 상위 5개 기사를 추천
    return recommended_articles[:5]


# 대화 기록을 바탕으로 한 AI 추천 함수
def get_conversation_based_recommendations(conversation_history, articles_data, current_article_id=None):
    """
    대화 기록을 바탕으로 AI가 실제 뉴스 데이터에서 기사를 추천하는 함수
    current_article_id: 현재 보고 있는 기사의 ID (제외용)
    """
    try:
        if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
            return "API 키가 설정되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정하거나 setup_api_key.sh를 실행하세요."
        
        # 대화 기록을 텍스트로 변환 (conversation_history 사용)
        conversation_text = ""
        for turn in conversation_history:
            if turn["role"] == "system":
                conversation_text += f"[시스템]: {turn['message']}\n\n"
            elif turn["role"] == "user":
                conversation_text += f"사용자: {turn['message']}\n"
            else:  # bot
                conversation_text += f"AI: {turn['message']}\n"
        
        # 실제 뉴스 데이터에서 샘플 기사들 준비 (이미 로드 시점에 50개로 제한됨)
        sample_articles = articles_data
        
        # 현재 기사가 있으면 추천 풀에서 제외 (ID 기준)
        if current_article_id:
            sample_articles = [article for article in sample_articles if str(article['id']) != str(current_article_id)]
        
        articles_info = ""
        for i, article in enumerate(sample_articles):
            body = article.get('content', '').replace('\n', ' ') #토큰절약을 위해
            articles_info += f"{article['id']}: {article['title']} | 내용: {body}\n"
        
        # 현재 기사 정보 추가
        current_article_info = ""
        if current_article_id:
            current_article = next((a for a in articles_data if str(a['id']) == str(current_article_id)), None)
            if current_article:
                current_body = current_article.get('content', '').replace('\n', ' ')
                current_article_info = f"""
현재 논의 중인 기사:
제목: {current_article['title']}
내용: {current_body}
"""

        # AI에게 추천 요청 (대화 기록 활용)
        prompt = f"""토론 기반 뉴스 추천 엔진입니다.

{current_article_info}
토론: {conversation_text}

사용 가능한 기사:
{articles_info}

현재 기사와 토론 맥락을 고려하여 관련 기사 5개를 추천하세요. 단 현재 논의 중인 기사는 제외해주세요.

형식:
1. 제목 (ID: X)
http://localhost:5002/article/X - 추천 이유
2. 제목 (ID: Y)  
http://localhost:5002/article/Y - 추천 이유
3. 제목 (ID: Z)
http://localhost:5002/article/Z - 추천 이유
4. 제목 (ID: A)
http://localhost:5002/article/A - 추천 이유
5. 제목 (ID: B)
http://localhost:5002/article/B - 추천 이유
"""
        
        # 프롬프트 내용 출력 (디버깅용)
        print("=" * 80)
        print("GPT에게 보내는 프롬프트:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "당신은 토론 기반 뉴스 추천 엔진입니다. 사용자의 토론 내용을 분석하여 제공된 실제 뉴스 데이터에서 토론을 심화할 수 있는 관련 기사들을 선택하여 추천해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_recommendation = response['choices'][0]['message']['content'].strip()
        return ai_recommendation
        
    except Exception as e:
        return f"AI 추천 생성 중 오류 발생: {str(e)}"


# 주요 단어의 의미를 GPT API를 이용해 가져오는 함수
def get_word_definitions(keywords):
    word_definitions = {}
    for word in keywords:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # 올바른 모델 이름
                messages=[
                    {"role": "system", "content": "단어의 정의를 한 줄로 간략하게 설명해줘"},
                    {"role": "user", "content": f"{word}"}
                ],
                max_tokens=100
            )
            definition = response['choices'][0]['message']['content'].strip()
            word_definitions[word] = definition
        except Exception as e:
            word_definitions[word] = f"Error: {str(e)}"
    return word_definitions


#50개 로드
articles_data = load_articles()

# SentenceTransformer 모델 로드
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# 요약 캐시 초기화 (성능 향상을 위해) - 제목 기반 추천으로 변경하여 비활성화
# summary_cache = load_summary_cache()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/article_list')
def article_list():
    return render_template('article_list.html', articles=articles_data[0:10])


@app.route('/article/<int:article_id>', methods=['GET', 'POST'])
def article(article_id):
    # articles_data에서 article_id에 맞는 기사를 가져옵니다.
    article = next((a for a in articles_data if a['id'] == article_id), None)

    if article is None:
        return "Article not found", 404

    try:
        # word_definitions = get_word_definitions(article['filtered_keywords'])  # 주요 단어 정의 가져오기 (주석처리)
        word_definitions = {}  # 단어 정의 비활성화 (빈 딕셔너리로 설정)
        # summary = get_summary(article['text'])  # 기사 요약 가져오기 (제목 기반 추천으로 변경하여 비활성화)
        summary = "제목 기반 추천 시스템으로 변경되어 요약 기능이 비활성화되었습니다."
        
        # 4컷 만화 생성 함수 호출 (주석처리)
        # image_url = generate_four_panel_comic(summary)
        image_url = None
    except Exception as e:
        print(f"API 호출 에러: {e}")
        # API 호출 실패 시 기본값 사용
        word_definitions = {}  # 빈 딕셔너리로 설정
        summary = "요약을 불러올 수 없습니다. API 키를 확인해주세요."
        image_url = None

    search_definition = None
    search_word = None

    if request.method == 'POST':
        search_query = request.form.get('search')
        if search_query:
            search_definition = get_word_definitions([search_query]).get(search_query)
            search_word = search_query  # 검색된 단어 저장

    # 클릭한 기사와 유사한 기사 추천하기 (요약 기반 추천 시스템 사용)
    try:
        recommended_articles = get_text_based_recommendations(article_id, articles_data, sentence_model)
    except Exception as e:
        print(f"추천 시스템 에러: {e}")
        recommended_articles = []

    return render_template('article.html', article=article, word_definitions=word_definitions,
                           summary=summary, search_definition=search_definition, search_word=search_word, recommended_articles=recommended_articles, articles=articles_data, image_url=image_url)


@app.route('/get_ai_recommendations', methods=['POST'])
def get_ai_recommendations():
    """대화 기록을 바탕으로 AI 추천을 생성하는 API"""
    try:
        data = request.get_json()
        conversation = data.get('conversation', [])
        current_article_id = data.get('current_article_id')
        
        if not conversation:
            return jsonify({"error": "대화 기록이 없습니다"}), 400
        
        # AI 추천 생성 (ID 직접 전달)
        ai_recommendation = get_conversation_based_recommendations(
            conversation, articles_data, current_article_id
        )
        
        return jsonify({"recommendation": ai_recommendation})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_cosine_recommendations', methods=['POST'])
def get_cosine_recommendations():
    """코사인 유사도 기반 추천을 생성하는 API"""
    try:
        data = request.get_json()
        article_id = data.get('article_id')
        
        if article_id is None:
            return jsonify({"error": "기사 ID가 필요합니다"}), 400
        
        # 코사인 유사도 기반 추천 생성
        recommended_articles = get_text_based_recommendations(article_id, articles_data, sentence_model)
        
        # 결과를 제목과 유사도 점수로 변환
        recommendations = []
        for idx, score in recommended_articles:
            recommendations.append((articles_data[idx]['title'], float(score)))
        
        return jsonify({"recommendations": recommendations})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5002)
