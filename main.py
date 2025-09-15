from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
import ast
import subprocess
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# from konlpy.tag import Okt  # 사용하지 않음
import numpy as np

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

def load_summary_cache():
    """캐시 파일에서 요약 캐시를 불러옵니다."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                print(f"캐시 로드 완료: {len(cache)}개 요약")
                return cache
    except Exception as e:
        print(f"캐시 로드 실패: {e}")
    return {}

def save_summary_cache(cache):
    """요약 캐시를 파일에 저장합니다."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
            print(f"캐시 저장 완료: {len(cache)}개 요약")
    except Exception as e:
        print(f"캐시 저장 실패: {e}")

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
    df = pd.read_csv('data/data.csv')  # 데이터 파일 읽기
    articles = []

    # iterrows()는 index와 row를 반환합니다.
    for index, row in df.iterrows():
        article = {
            'id': index,  # 'id'를 index로 설정
            'title': row['title'],
            'text': row['text'],
            'processed_text': row['processed_text'],
            'tfidf_keywords': safe_literal_eval(row['tfidf_keywords']),
            'tfidf_scores': safe_literal_eval(row['tfidf_scores']),
            'keybert_keywords': safe_literal_eval(row['keybert_keywords']),
            'keybert_scores': safe_literal_eval(row['keybert_scores']),
            'filtered_keywords': safe_literal_eval(row['filtered_keywords']),
        }
        articles.append(article)

    return articles


# `ast.literal_eval`을 안전하게 처리하는 함수
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return []



# def get_title_embeddings(articles_data, word2vec_model):
#     title_embeddings = []
#     for article in articles_data:
#         title = article['title']
#         words = title.split()  # 제목을 공백 기준으로 단어 분리
#         title_vector = []

#         for word in words:
#             if word in word2vec_model.wv:
#                 title_vector.append(word2vec_model.wv[word])

#         # 제목에 대한 벡터가 있으면 평균 벡터를 반환
#         if title_vector:
#             title_embeddings.append(np.mean(title_vector, axis=0))
#         else:
#             title_embeddings.append(np.zeros(word2vec_model.vector_size))  # 벡터가 없으면 0 벡터 반환

#     return title_embeddings


def get_summary_embeddings(articles_data, word2vec_model, summary_cache=None):
    """
    기사 요약을 기반으로 임베딩을 생성하는 함수
    summary_cache: 기사 ID를 키로 하고 요약을 값으로 하는 딕셔너리 (캐싱용)
    """
    summary_embeddings = []
    if summary_cache is None:
        summary_cache = {}
    
    for article in articles_data:
        article_id = article['id']
        
        # 캐시에서 요약을 찾거나 새로 생성
        if article_id in summary_cache:
            summary = summary_cache[article_id]
        else:
            summary = get_summary(article['text'])
            summary_cache[article_id] = summary
        
        # 요약 텍스트를 단어로 분리
        words = summary.split()
        summary_vector = []

        for word in words:
            if word in word2vec_model.wv:
                summary_vector.append(word2vec_model.wv[word])

        # 요약에 대한 벡터가 있으면 평균 벡터를 반환
        if summary_vector:
            summary_embeddings.append(np.mean(summary_vector, axis=0))
        else:
            summary_embeddings.append(np.zeros(word2vec_model.vector_size))  # 벡터가 없으면 0 벡터 반환

    return summary_embeddings, summary_cache


# def get_recommendations(clicked_article_id, articles, word2vec_model):
#     # 클릭한 기사 가져오기
#     clicked_article = next((a for a in articles if a['id'] == clicked_article_id), None)

#     if clicked_article is None:
#         return []  # 클릭한 기사를 찾을 수 없으면 빈 리스트 반환

#     # 클릭한 기사의 요약을 생성하고 임베딩 계산
#     clicked_summary = get_summary(clicked_article['text'])
#     clicked_summary_words = clicked_summary.split()
#     clicked_embedding = np.mean([word2vec_model.wv[word] for word in clicked_summary_words if word in word2vec_model.wv], axis=0)

#     if np.isnan(clicked_embedding).any() or len(clicked_summary_words) == 0:
#         clicked_embedding = np.zeros(word2vec_model.vector_size)  # 임베딩이 없으면 0 벡터로 설정

#     # 요약 임베딩 계산 (전체 기사 목록에 대해)
#     summary_embeddings = get_summary_embeddings(articles, word2vec_model)

#     # 클릭한 기사와 다른 기사들의 코사인 유사도 계산
#     similarities = cosine_similarity([clicked_embedding], summary_embeddings).flatten()

#     # 유사도가 높은 순으로 기사 추천 (자기 자신은 제외)
#     recommended_articles = sorted(
#         [(i, sim) for i, sim in enumerate(similarities) if articles[i]['id'] != clicked_article_id],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     # 유효한 인덱스만 필터링
#     valid_articles = [article for article in recommended_articles if article[0] < len(articles)]

#     # 상위 5개 기사를 추천
#     return valid_articles[:5]


def get_summary_only_recommendations(clicked_article_id, articles, word2vec_model, summary_cache=None):
    """
    요약만을 사용한 추천 시스템 (순수 요약 기반)
    """
    # 클릭한 기사 가져오기
    clicked_article = next((a for a in articles if a['id'] == clicked_article_id), None)

    if clicked_article is None:
        return []  # 클릭한 기사를 찾을 수 없으면 빈 리스트 반환

    # 요약 기반 임베딩 (캐시 사용)
    if summary_cache is None:
        summary_cache = {}
    
    if clicked_article_id in summary_cache:
        clicked_summary = summary_cache[clicked_article_id]
    else:
        clicked_summary = get_summary(clicked_article['text'])
        summary_cache[clicked_article_id] = clicked_summary
    
    clicked_summary_words = clicked_summary.split()
    clicked_summary_embedding = np.mean([word2vec_model.wv[word] for word in clicked_summary_words if word in word2vec_model.wv], axis=0)
    if np.isnan(clicked_summary_embedding).any() or len(clicked_summary_words) == 0:
        clicked_summary_embedding = np.zeros(word2vec_model.vector_size)

    # 성능 최적화: 전체 기사 대신 샘플 기사들만 사용 (처음 20개 기사)
    sample_articles = articles[:20]  # 처음 20개 기사만 사용
    summary_embeddings, updated_cache = get_summary_embeddings(sample_articles, word2vec_model, summary_cache)

    # 클릭한 기사와 다른 기사들의 코사인 유사도 계산
    similarities = cosine_similarity([clicked_summary_embedding], summary_embeddings).flatten()

    # 유사도가 높은 순으로 기사 추천 (자기 자신은 제외)
    recommended_articles = sorted(
        [(i, sim) for i, sim in enumerate(similarities) if sample_articles[i]['id'] != clicked_article_id],
        key=lambda x: x[1],
        reverse=True
    )

    # 유효한 인덱스만 필터링
    valid_articles = [article for article in recommended_articles if article[0] < len(sample_articles)]

    # 상위 5개 기사를 추천
    return valid_articles[:5]


# def get_hybrid_recommendations(clicked_article_id, articles, word2vec_model, summary_cache=None):
#     """
#     제목과 요약을 모두 고려한 하이브리드 추천 시스템
#     """
#     # 클릭한 기사 가져오기
#     clicked_article = next((a for a in articles if a['id'] == clicked_article_id), None)

#     if clicked_article is None:
#         return []  # 클릭한 기사를 찾을 수 없으면 빈 리스트 반환

#     # 제목 기반 임베딩
#     clicked_title = clicked_article['title']
#     clicked_title_embedding = np.mean([word2vec_model.wv[word] for word in clicked_title.split() if word in word2vec_model.wv], axis=0)
#     if np.isnan(clicked_title_embedding).any():
#         clicked_title_embedding = np.zeros(word2vec_model.vector_size)

#     # 요약 기반 임베딩 (캐시 사용)
#     if summary_cache is None:
#         summary_cache = {}
    
#     if clicked_article_id in summary_cache:
#         clicked_summary = summary_cache[clicked_article_id]
#     else:
#         clicked_summary = get_summary(clicked_article['text'])
#         summary_cache[clicked_article_id] = clicked_summary
    
#     clicked_summary_words = clicked_summary.split()
#     clicked_summary_embedding = np.mean([word2vec_model.wv[word] for word in clicked_summary_words if word in word2vec_model.wv], axis=0)
#     if np.isnan(clicked_summary_embedding).any() or len(clicked_summary_words) == 0:
#         clicked_summary_embedding = np.zeros(word2vec_model.vector_size)

#     # 제목과 요약 임베딩을 결합 (가중 평균: 요약 70%, 제목 30%)
#     clicked_combined_embedding = 0.7 * clicked_summary_embedding + 0.3 * clicked_title_embedding

#     # 전체 기사들의 하이브리드 임베딩 계산
#     title_embeddings = get_title_embeddings(articles, word2vec_model)
#     summary_embeddings, updated_cache = get_summary_embeddings(articles, word2vec_model, summary_cache)
    
#     combined_embeddings = []
#     for i in range(len(articles)):
#         combined_embedding = 0.7 * summary_embeddings[i] + 0.3 * title_embeddings[i]
#         combined_embeddings.append(combined_embedding)

#     # 클릭한 기사와 다른 기사들의 코사인 유사도 계산
#     similarities = cosine_similarity([clicked_combined_embedding], combined_embeddings).flatten()

#     # 유사도가 높은 순으로 기사 추천 (자기 자신은 제외)
#     recommended_articles = sorted(
#         [(i, sim) for i, sim in enumerate(similarities) if articles[i]['id'] != clicked_article_id],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     # 유효한 인덱스만 필터링
#     valid_articles = [article for article in recommended_articles if article[0] < len(articles)]

#     # 상위 5개 기사를 추천
#     return valid_articles[:5]

# 대화 기록을 바탕으로 한 AI 추천 함수
def get_conversation_based_recommendations(conversation_history, articles_data, num_recommendations=5):
    """
    대화 기록을 바탕으로 AI가 실제 뉴스 데이터에서 기사를 추천하는 함수
    """
    try:
        if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
            return "API 키가 설정되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정하거나 setup_api_key.sh를 실행하세요."
        
        # 대화 기록을 텍스트로 변환
        conversation_text = ""
        for turn in conversation_history:
            role = "사용자" if turn["role"] == "user" else "AI"
            conversation_text += f"{role}: {turn['message']}\n"
        
        # 현재 기사 정보 가져오기 (첫 번째 대화에서 기사 정보가 있을 경우)
        article_summary = ""
        if len(conversation_history) > 0:
            # 첫 번째 AI 메시지에서 기사 정보 추출 시도
            first_ai_message = None
            for turn in conversation_history:
                if turn["role"] == "assistant":
                    first_ai_message = turn["message"]
                    break
            
            if first_ai_message and "기사 요약:" in first_ai_message:
                # 기사 정보가 포함된 경우 요약 추출
                lines = first_ai_message.split('\n')
                for line in lines:
                    if "기사 요약:" in line:
                        article_summary = line.replace("기사 요약:", "").strip()
                        break
        
        # 실제 뉴스 데이터에서 샘플 기사들 준비 (처음 100개 기사)
        sample_articles = articles_data[:100]
        articles_info = ""
        for i, article in enumerate(sample_articles):
            articles_info += f"{i+1}. ID: {article['id']}, 제목: {article['title']}\n"
        
        # AI에게 추천 요청
        prompt = f"""
당신은 토론 기반 뉴스 추천 엔진입니다.  
다음은 사용자가 읽은 기사 요약과 챗봇과의 토론 기록입니다.  

[기사 요약]  
{article_summary if article_summary else "기사 요약 정보가 없습니다."}  

[토론 기록]  
{conversation_text}  

[사용 가능한 뉴스 데이터]
{articles_info}

위 토론에서 사용자가 특히 관심을 보인 쟁점, 질문, 의견을 분석하세요.  
그 후, 사용 가능한 뉴스 데이터에서 토론을 심화할 수 있는 기사 5개를 선택하여 추천하세요.  

출력 형식:  
1. 기사 제목 (ID: X)
http://localhost:5002/article/X – 왜 추천하는지 (토론 맥락과 연결)  
2. 기사 제목 (ID: Y)
http://localhost:5002/article/Y - 왜 추천하는지  
3. 기사 제목 (ID: Z)
http://localhost:5002/article/Z - 왜 추천하는지  
4. 기사 제목 (ID: A)
http://localhost:5002/article/A - 왜 추천하는지  
5. 기사 제목 (ID: B)
http://localhost:5002/article/B - 왜 추천하는지  

반드시 위의 사용 가능한 뉴스 데이터에서만 선택하세요.
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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


# 기사를 요약하는 함수
def get_summary(content):
    try:
        if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
            return "API 키가 설정되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정하거나 setup_api_key.sh를 실행하세요."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 올바른 모델 이름
            messages=[
                {"role": "system", "content": "너는 친절하게 답변해주는 비서야. 다음의 기사를 적절하게 한 문장 내로 요약해줘."},
                {"role": "user", "content": content}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"요약 생성 실패: {str(e)}"
    

def generate_four_panel_comic(article_summary):
    """
    요약 토대로 이미지를 생성하는 함수.

    Returns:
        str: 생성된 이미지의 URL (오류 시 에러 메시지 반환)
    """
    
    # 2. 만화 프롬프트 생성
    prompt = (
        f"Create an simple, small image based on summary (no text involved in the image)'{article_summary}': "
    )

    # 3. DALL·E API 호출하여 이미지 생성
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']
    except Exception as e:
        return f"이미지 생성 중 오류 발생: {str(e)}"


articles_data = load_articles()

# Word2Vec 모델 로드 (이미 학습된 모델을 사용)
word2vec_model = Word2Vec.load("utils/word2vec_model.model")  # 학습된 모델 경로로 변경

# 요약 캐시 초기화 (성능 향상을 위해)
summary_cache = load_summary_cache()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/article_list')
def article_list():
    return render_template('article_list.html', articles=articles_data[6310:6319])


@app.route('/article/<int:article_id>', methods=['GET', 'POST'])
def article(article_id):
    # articles_data에서 article_id에 맞는 기사를 가져옵니다.
    article = next((a for a in articles_data if a['id'] == article_id), None)

    if article is None:
        return "Article not found", 404

    try:
        # word_definitions = get_word_definitions(article['filtered_keywords'])  # 주요 단어 정의 가져오기 (주석처리)
        word_definitions = {}  # 단어 정의 비활성화 (빈 딕셔너리로 설정)
        summary = get_summary(article['text'])  # 기사 요약 가져오기
        
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
        global summary_cache
        recommended_articles = get_summary_only_recommendations(article_id, articles_data, word2vec_model, summary_cache)
        # 캐시 업데이트 후 파일에 저장
        save_summary_cache(summary_cache)
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
        
        if not conversation:
            return jsonify({"error": "대화 기록이 없습니다"}), 400
        
        # AI 추천 생성
        ai_recommendation = get_conversation_based_recommendations(conversation, articles_data)
        
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
        global summary_cache
        recommended_articles = get_summary_only_recommendations(article_id, articles_data, word2vec_model, summary_cache)
        # 캐시 업데이트 후 파일에 저장
        save_summary_cache(summary_cache)
        
        # 결과를 제목과 유사도 점수로 변환
        recommendations = []
        for idx, score in recommended_articles:
            recommendations.append((articles_data[idx]['title'], float(score)))
        
        return jsonify({"recommendations": recommendations})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5002)
