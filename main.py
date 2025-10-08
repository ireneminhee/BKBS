from flask import Flask, render_template, request, jsonify
import openai
from openai import OpenAI
import pandas as pd
import ast
import subprocess
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# from konlpy.tag import Okt  # 사용하지 않음
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.evaluation import aggregate_diversity_report
import base64

app = Flask(__name__)

# 캐시 초기화 curl -X POST http://localhost:5002/clear_cache

# =============================================================================
# API 기능 제어 설정 (한번에 켜고 끌 수 있음)
# =============================================================================
# True: API 기능 활성화, False: API 기능 비활성화
ENABLE_ARTICLE_SUMMARY = True      # 기사 요약 기능
ENABLE_FOUR_PANEL_COMIC = True     # 네컷만화 생성 기능  
ENABLE_WORD_DEFINITIONS = True    # 단어 정의 기능
ENABLE_AI_RECOMMENDATIONS = True  # AI 기반 추천 기능
ENABLE_DIVERSITY_METRICS = True   # 다양성 지표 계산 기능

# API 기능이 비활성화되었을 때 표시할 메시지
DISABLED_MESSAGE = "⚠️ 이 기능은 현재 비활성화되어 있습니다."

# 사용법:
# 1. 모든 API 기능을 끄려면: 모든 변수를 False로 설정
# 2. 특정 기능만 끄려면: 해당 변수만 False로 설정
# 3. 모든 기능을 켜려면: 모든 변수를 True로 설정
# 4. 설정 변경 후 서버를 재시작하면 적용됩니다.
# =============================================================================

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

# 캐시 파일 경로들
SUMMARY_CACHE_FILE = "summary_cache.pkl"
COMIC_CACHE_FILE = "comic_cache.pkl"
WORD_DEFINITION_CACHE_FILE = "word_definition_cache.pkl"

# API 사용량 추적
API_USAGE_TRACKER = {
    'summary_calls': 0,
    'comic_calls': 0,
    'word_definition_calls': 0,
    'ai_recommendation_calls': 0
}

def load_cache(cache_file):
    """캐시 파일에서 데이터를 불러옵니다."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                print(f"캐시 로드 완료: {cache_file} - {len(cache)}개 항목")
                return cache
    except Exception as e:
        print(f"캐시 로드 실패 ({cache_file}): {e}")
    return {}

def save_cache(cache, cache_file):
    """캐시 데이터를 파일에 저장합니다."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
            print(f"캐시 저장 완료: {cache_file} - {len(cache)}개 항목")
    except Exception as e:
        print(f"캐시 저장 실패 ({cache_file}): {e}")

def get_cache_key(content, prefix=""):
    """콘텐츠를 기반으로 캐시 키를 생성합니다."""
    import hashlib
    content_str = str(content) + prefix
    return hashlib.md5(content_str.encode()).hexdigest()

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
    df = pd.read_csv('./data/news_data_1.csv') # 데이터 파일 읽기
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
            'text': row['text'],
            'link': row['link'],  # 원본 링크 추가
            'preprocessed_text': row['preprocessed_text']
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
        # title + text 결합
        preprocessed_text = article['title'] + " " + article['text']
        texts.append(preprocessed_text)
    
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
    clicked_text = clicked_article['title'] + " " + clicked_article['text']
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

    return recommended_articles[:10]


# 대화 기록을 바탕으로 한 AI 추천 함수
def get_conversation_based_recommendations(conversation_history, articles_data, current_article_id=None):
    """
    대화 기록을 바탕으로 AI가 실제 뉴스 데이터에서 기사를 추천하는 함수
    current_article_id: 현재 보고 있는 기사의 ID (제외용)
    """
    try:
        # API 기능이 비활성화된 경우
        if not ENABLE_AI_RECOMMENDATIONS:
            return DISABLED_MESSAGE
        
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
            body = article.get('text', '').replace('\n', ' ') #토큰절약을 위해
            articles_info += f"{article['id']}: {article['title']} | 내용: {body}\n"
        
        # 현재 기사 정보 가져오기
        current_article = None
        if current_article_id:
            current_article = next((a for a in articles_data if str(a['id']) == str(current_article_id)), None)
        
        current_article_info = ""
        if current_article:
            # 백슬래시 문제 해결을 위해 변수에 저장
            article_text = current_article['text'].replace('\n', ' ')
            current_article_info = f"""
현재 기사:
제목: {current_article['title']}
내용: {article_text}
"""
        
        # AI에게 추천 요청 (대화 기록 활용)
        prompt = f"""토론 기반 뉴스 추천 엔진입니다.{current_article_info}

토론: {conversation_text}

사용 가능한 기사:
{articles_info}

현재 기사와 토론 맥락을 고려하여 관련 기사 10개를 추천하세요. 단 현재 논의 중인 기사는 제외해주세요.

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
6. 제목 (ID: C)
http://localhost:5002/article/C - 추천 이유
7. 제목 (ID: D)
http://localhost:5002/article/D - 추천 이유
8. 제목 (ID: E)
http://localhost:5002/article/E - 추천 이유
9. 제목 (ID: F)
http://localhost:5002/article/F - 추천 이유
10. 제목 (ID: G)
http://localhost:5002/article/G - 추천 이유
"""
        
        # 프롬프트 내용을 터미널에 출력
        print("=" * 80)
        print("🤖 AI 추천 프롬프트:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        API_USAGE_TRACKER['ai_recommendation_calls'] += 1
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 토론 기반 뉴스 추천 엔진입니다. 사용자의 토론 내용을 분석하여 제공된 실제 뉴스 데이터에서 토론을 심화할 수 있는 관련 기사들을 선택하여 추천해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_recommendation = response.choices[0].message.content.strip()
        
        return ai_recommendation
        
    except Exception as e:
        return f"AI 추천 생성 중 오류 발생: {str(e)}"


# AI 추천 결과에서 기사 ID들을 추출하는 함수
def parse_ai_recommendation_ids(ai_recommendation_text):
    """
    AI 추천 텍스트에서 추천된 기사의 ID들을 추출
    Streamlit에서 생성된 하이퍼링크에서 ID 추출
    """
    import re
    
    found_ids = set()
    
    # Streamlit에서 생성된 하이퍼링크 패턴: [제목](http://localhost:5002/article/ID) - 설명
    # 또는 URL 패턴: http://localhost:5002/article/ID
    id_patterns = [
        r'\[.*?\]\(http://localhost:5002/article/(\d+)\)',  # 마크다운 링크 패턴
        r'http://localhost:5002/article/(\d+)'  # URL 패턴
    ]
    
    for pattern in id_patterns:
        matches = re.findall(pattern, ai_recommendation_text)
        for match in matches:
            found_ids.add(int(match))
    
    # 디버깅 로그
    print(f"추출된 기사 IDs: {list(found_ids)}")
    
    # 만약 링크에서 ID를 찾지 못했다면 원본 AI 추천 텍스트에서 URL 패턴 다시 확인
    if not found_ids:
        # 원본 텍스트에서 직접 URL 패턴 찾기
        url_pattern = r'http://localhost:5002/article/(\d+)'
        url_matches = re.findall(url_pattern, ai_recommendation_text)
        # 순서를 유지하기 위해 set 대신 list 사용
        for match in url_matches:
            found_ids.add(int(match))
        print(f"URL 패턴으로 재추출된 IDs: {list(found_ids)}")
    
    # 순서 유지를 위해 발견된 순서대로 반환 (첫 번째 발견 순서)
    final_ids = []
    url_pattern = r'http://localhost:5002/article/(\d+)'
    url_matches = re.findall(url_pattern, ai_recommendation_text)
    for match in url_matches:
        if int(match) not in final_ids:
            final_ids.append(int(match))
    return final_ids if final_ids else list(found_ids)


# AI 추천 기사들의 diversity metrics을 계산하는 함수
def calculate_ai_recommendation_diversity(recommendation_ids, articles_data, sentence_model):
    """
    AI 추천 기사들의 diversity metrics 계산
    """
    try:
        # API 기능이 비활성화된 경우
        if not ENABLE_DIVERSITY_METRICS:
            return {"error": DISABLED_MESSAGE}
        
        if len(recommendation_ids) < 2:
            return {"error": "추천된 기사가 2개 미만입니다. Diversity 계산을 위해 최소 2개 이상 필요합니다."}
        
        # 추천된 기사들 찾기
        recommended_articles = []
        for article in articles_data:
            if article['id'] in recommendation_ids:
                recommended_articles.append(article)
        
        if len(recommended_articles) < 2:
            return {"error": "추천된 기사를 데이터에서 찾을 수 없습니다."}
        
        # DataFrame 생성 (AI 추천은 모든 기사를 하나의 seed로 처리)
        import pandas as pd
        df_data = []
        for article in recommended_articles:
            df_data.append({
                'seed_article_id': 'ai_recommendation_group',  # 모든 AI 추천을 하나의 그룹으로 처리
                'title': article['title'],
                'context': article['text'],
                'source': article.get('source', 'Unknown')  # source 컬럼이 없다면 기본값
            })
        
        df = pd.DataFrame(df_data)
        
        # diversity metrics 계산 (AI 추천 그룹용)
        print("=== AI 추천 다양성 분석 시작 ===")
        print(f"추천 기사 수: {len(df)}")
        print(f"추천 기사 출처들: {df['source'].tolist()}")
        
        diversity_report = aggregate_diversity_report(df, sentence_model)
        
        print("=== AI 추천 다양성 분석 종료 ===\n")
        
        # JSON 직렬화 가능한 형태로 변환 (DataFrame 제외)
        json_safe_report = {
            'ild': diversity_report['ild'],
            'cgi': diversity_report['cgi'],
            'per_seed_summary': {
                'total_seeds': len(diversity_report['per_seed_df']),
                'avg_ild': float(diversity_report['per_seed_df']['ild'].mean()),
                'avg_cgi': float(diversity_report['per_seed_df']['cgi'].mean()),
                'recommendation_counts': diversity_report['per_seed_df']['n_recommendations'].tolist()
            }
        }
        
        return json_safe_report
        
    except Exception as e:
        return {"error": f"Diversity 계산 중 오류 발생: {str(e)}"}


# 주요 단어의 의미를 GPT API를 이용해 가져오는 함수
def get_word_definitions(keywords):
    # API 기능이 비활성화된 경우
    if not ENABLE_WORD_DEFINITIONS:
        return {word: DISABLED_MESSAGE for word in keywords}
    
    word_definitions = {}
    client = OpenAI(api_key=openai.api_key)
    
    for word in keywords:
        # 캐시에서 먼저 확인
        cache_key = get_cache_key(word, "word_def")
        if cache_key in word_definition_cache:
            word_definitions[word] = word_definition_cache[cache_key]
            print(f"📋 캐시에서 단어 정의 로드: {word}")
            continue
            
        try:
            API_USAGE_TRACKER['word_definition_calls'] += 1
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 올바른 모델 이름
                messages=[
                    {"role": "system", "content": "단어의 정의를 한 줄로 간략하게 설명해줘"},
                    {"role": "user", "content": f"{word}"}
                ],
                max_tokens=100
            )
            definition = response.choices[0].message.content.strip()
            word_definitions[word] = definition
            
            # 캐시에 저장
            word_definition_cache[cache_key] = definition
            save_cache(word_definition_cache, WORD_DEFINITION_CACHE_FILE)
            print(f"💾 단어 정의 캐시 저장: {word}")
            
        except Exception as e:
            word_definitions[word] = f"Error: {str(e)}"
    
    return word_definitions

# 기사를 요약하는 함수
def get_summary(content):
    # API 기능이 비활성화된 경우
    if not ENABLE_ARTICLE_SUMMARY:
        return DISABLED_MESSAGE
    
    # 캐시에서 먼저 확인
    cache_key = get_cache_key(content, "summary")
    if cache_key in summary_cache:
        print(f"📋 캐시에서 요약 로드")
        return summary_cache[cache_key]
    
    try:
        if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
            return "API 키가 설정되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정하거나 setup_api_key.sh를 실행하세요."
        
        API_USAGE_TRACKER['summary_calls'] += 1
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 올바른 모델 이름
            messages=[
                {"role": "system", "content": "너는 친절하게 답변해주는 비서야. 다음의 기사를 적절하게 한 문장으로 요약해줘."},
                {"role": "user", "content": content}
            ],
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
        
        # 캐시에 저장
        summary_cache[cache_key] = summary
        save_cache(summary_cache, SUMMARY_CACHE_FILE)
        print(f"💾 요약 캐시 저장")
        
        return summary
    except Exception as e:
        return f"요약 생성 실패: {str(e)}"


def generate_four_panel_comic(article_summary):
    """
    기사 요약을 바탕으로 네컷 만화를 생성하는 함수.
    사용자가 한눈에 기사 내용을 파악할 수 있도록 시각적으로 표현합니다.

    Returns:
        str: 생성된 이미지의 URL (오류 시 에러 메시지 반환)
    """
    
    # API 기능이 비활성화된 경우
    if not ENABLE_FOUR_PANEL_COMIC:
        return DISABLED_MESSAGE
    
    # 캐시에서 먼저 확인
    cache_key = get_cache_key(article_summary, "comic")
    if cache_key in comic_cache:
        print(f"📋 캐시에서 네컷만화 로드")
        return comic_cache[cache_key]
    
    print(f"🎨 네컷만화 생성 시작 - 요약: {article_summary[:50]}...")
    
    # 네컷 만화 프롬프트 생성
    prompt = f"""Create a 4-panel comic strip that visually tells the story from this news article summary: '{article_summary}'

Panel structure:
- Panel 1: Initial situation or context (what happened first)
- Panel 2: Development or problem arising (what unfolded)
- Panel 3: Key event or turning point (main incident)
- Panel 4: Resolution or current status (how it ended or current state)

Style requirements:
- Clean, simple cartoon style
- NO TEXT, NO SPEECH BUBBLES, NO WORDS anywhere in the image
- Each panel should be clearly distinct
- Use expressive characters and clear visual storytelling
- Professional news illustration style
- High contrast and clear composition
- Suitable for all audiences
- Pure visual storytelling without any written language

The comic should help readers quickly understand the main points of the news story through visual narrative only, without any text or words."""

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=openai.api_key)
    
    # DALL-E 3 모델을 사용한 이미지 생성
    try:
        print("🔄 DALL-E 3 API 호출 중...")
        API_USAGE_TRACKER['comic_calls'] += 1
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1792x1024",
            quality="hd",
            style="natural"
        )
        
        print("✅ 이미지 생성 성공!")
        image_url = response.data[0].url
        print(f"🖼️ 이미지 URL: {image_url[:50]}...")
        
        # 캐시에 저장
        comic_cache[cache_key] = image_url
        save_cache(comic_cache, COMIC_CACHE_FILE)
        print(f"💾 네컷만화 캐시 저장")
        
        # 이미지 URL 반환
        return image_url
            
    except Exception as e:
        error_msg = f"네컷 만화 생성 중 오류 발생: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


#50개 로드
articles_data = load_articles()

# SentenceTransformer 모델 로드
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# 캐시 초기화 (성능 향상을 위해)
summary_cache = load_cache(SUMMARY_CACHE_FILE)
comic_cache = load_cache(COMIC_CACHE_FILE)
word_definition_cache = load_cache(WORD_DEFINITION_CACHE_FILE)

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
        # 기사 요약 생성
        print("📝 기사 요약 생성 중...")
        summary = get_summary(article['text'])  # 기사 본문 기반 요약 가져오기
        print(f"✅ 요약 생성 완료: {summary[:50]}...")
        
        # 4컷 만화 생성 함수 호출
        print("🎨 네컷만화 생성 시작...")
        image_url = generate_four_panel_comic(summary)
        print(f"🖼️ 이미지 URL 결과: {image_url[:50] if image_url else 'None'}...")
        
        # 주요 단어 정의 생성 (샘플 키워드 사용)
        print("🔤 주요 단어 정의 생성 중...")
        sample_keywords = ["정치", "경제", "사회", "기술", "환경"]  # 샘플 키워드
        word_definitions = get_word_definitions(sample_keywords)
        print(f"✅ 단어 정의 생성 완료: {len(word_definitions)}개")
        
    except Exception as e:
        print(f"❌ API 호출 에러: {e}")
        # API 호출 실패 시 기본값 사용
        word_definitions = {}  # 빈 딕셔너리로 설정
        
        # API 키 확인
        if not openai.api_key or openai.api_key == "YOUR_API_KEY_HERE":
            summary = "⚠️ OpenAI API 키가 설정되지 않았습니다. setup_api_key.sh를 실행하여 API 키를 설정해주세요."
            image_url = None
        else:
            summary = f"요약 생성 중 오류가 발생했습니다: {str(e)}"
            image_url = None

    search_definition = None
    search_word = None

    if request.method == 'POST':
        search_query = request.form.get('search')
        if search_query:
            search_definition = get_word_definitions([search_query]).get(search_query)
            search_word = search_query  # 검색된 단어 저장

    # 클릭한 기사와 유사한 기사 추천하기 
    try:
        recommended_articles = get_text_based_recommendations(article_id, articles_data, sentence_model)
        
        # 추천 결과 DataFrame 생성
        df_recs = pd.DataFrame([{
            'seed_article_id': article_id,
            'title': articles_data[idx]['title'],
            'context': articles_data[idx]['text'],
            'source': articles_data[idx]['source'],
            'article_id': articles_data[idx]['id'],  # 실제 기사 ID 추가
            'score': float(score)
        } for idx, score in recommended_articles])
        
        # diversity_metrics 계산 (API 기능이 활성화된 경우에만)
        if ENABLE_DIVERSITY_METRICS:
            print("=== 텍스트 기반 추천 다양성 분석 시작 ===")
            print(f"추천 기사 수: {len(df_recs)}")
            print(f"추천 기사 ID들: {df_recs['article_id'].tolist()}")
            print(f"추천 기사 출처들: {df_recs['source'].tolist()}")
            
            diversity_report = aggregate_diversity_report(df_recs, sentence_model)
            
            print("=== 텍스트 기반 추천 다양성 분석 종료 ===\n")

            diversity_metrics = {
                'ILD_mean': diversity_report['ild']['mean'],
                'ILD_std': diversity_report['ild']['std'],
                'ILD_ci_low': diversity_report['ild']['ci_low'],
                'ILD_ci_high': diversity_report['ild']['ci_high'],
                'CGI_mean': diversity_report['cgi']['mean'],
                'CGI_std': diversity_report['cgi']['std'],
                'CGI_ci_low': diversity_report['cgi']['ci_low'],
                'CGI_ci_high': diversity_report['cgi']['ci_high'],
            }
        else:
            # API 기능이 비활성화된 경우 기본값 설정
            diversity_metrics = {
                'ILD_mean': 0,
                'ILD_std': 0,
                'ILD_ci_low': 0,
                'ILD_ci_high': 0,
                'CGI_mean': 0,
                'CGI_std': 0,
                'CGI_ci_low': 0,
                'CGI_ci_high': 0,
            }

    except Exception as e:
        print(f"추천 시스템 에러: {e}")
        recommended_articles = []
        # 에러 발생 시 기본값 설정
        diversity_metrics = {
            'ILD_mean': 0,
            'ILD_std': 0,
            'ILD_ci_low': 0,
            'ILD_ci_high': 0,
            'CGI_mean': 0,
            'CGI_std': 0,
            'CGI_ci_low': 0,
            'CGI_ci_high': 0,
        }

    return render_template('article.html', article=article, word_definitions=word_definitions,
                           summary=summary, search_definition=search_definition, 
                           search_word=search_word, recommended_articles=recommended_articles, 
                           articles=articles_data, image_url=image_url,
                           diversity_metrics=diversity_metrics)


@app.route('/get_ai_recommendations', methods=['POST'])
def get_ai_recommendations():
    """대화 기록을 바탕으로 AI 추천을 생성하는 API"""
    try:
        # API 기능이 비활성화된 경우
        if not ENABLE_AI_RECOMMENDATIONS:
            return jsonify({"recommendation": DISABLED_MESSAGE})
        
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


@app.route('/get_ai_diversity_metrics', methods=['POST'])
def get_ai_diversity_metrics():
    """AI 추천 결과의 diversity metrics를 계산하는 API"""
    try:
        # API 기능이 비활성화된 경우
        if not ENABLE_DIVERSITY_METRICS:
            return jsonify({"error": DISABLED_MESSAGE})
        
        data = request.get_json()
        ai_recommendation_text = data.get('ai_recommendation', '')
        
        if not ai_recommendation_text:
            return jsonify({"error": "AI 추천 결과가 없습니다"}), 400
        
        # AI 추천 결과에서 기사 ID 추출
        recommendation_ids = parse_ai_recommendation_ids(ai_recommendation_text)
        
        if not recommendation_ids:
            return jsonify({"error": "추천 결과에서 기사 ID를 찾을 수 없습니다"}), 400
        
        # diversity metrics 계산
        diversity_report = calculate_ai_recommendation_diversity(
            recommendation_ids, articles_data, sentence_model
        )
        
        # 결과에 추출된 ID 추가
        diversity_report['extracted_ids'] = recommendation_ids
        
        return jsonify(diversity_report)
        
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


@app.route('/api_usage_stats', methods=['GET'])
def api_usage_stats():
    """API 사용량 통계를 반환하는 엔드포인트"""
    try:
        # 캐시 통계 추가
        cache_stats = {
            'summary_cache_size': len(summary_cache),
            'comic_cache_size': len(comic_cache),
            'word_definition_cache_size': len(word_definition_cache)
        }
        
        # 전체 통계
        stats = {
            'api_calls': API_USAGE_TRACKER.copy(),
            'cache_stats': cache_stats,
            'total_api_calls': sum(API_USAGE_TRACKER.values()),
            'cache_hit_potential': sum(cache_stats.values())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """캐시를 초기화하는 엔드포인트"""
    try:
        global summary_cache, comic_cache, word_definition_cache
        
        # 캐시 초기화
        summary_cache.clear()
        comic_cache.clear()
        word_definition_cache.clear()
        
        # 파일에서도 삭제
        for cache_file in [SUMMARY_CACHE_FILE, COMIC_CACHE_FILE, WORD_DEFINITION_CACHE_FILE]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        
        return jsonify({"message": "모든 캐시가 초기화되었습니다"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5002)