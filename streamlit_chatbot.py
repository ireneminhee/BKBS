import streamlit as st  # type: ignore
from utils.chatgpt_api import chatgpt_response
import requests
import json

# Streamlit 페이지 설정
st.set_page_config(page_title="신문기사 기반 토론 챗봇", layout="centered")
st.caption("사용자의 발언에 대한 AI의 반대 또는 다른 의견을 들어보세요.")

# 상태 초기화
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []  # 대화 저장
if "article" not in st.session_state:
    st.session_state["article"] = None  # 기사 데이터 (필요시 확장 가능)
if "initial_statement_done" not in st.session_state:
    st.session_state["initial_statement_done"] = True  # 첫 발언 여부 (주제 제거로 기본값 True)

# 쿼리 매개변수에서 기사 데이터를 세션 상태에 저장
query_params = st.query_params
if query_params.get("article_id") and query_params.get("title") and query_params.get("text"):
    st.session_state["article"] = {
        "id": query_params.get("article_id"),
        "title": query_params.get("title"),
        "text": query_params.get("text")
    }

# 기사 정보 출력
article = st.session_state.get("article")
if article:
    st.write(f"Article ID: {article['id']}")
    st.write(f"Title: {article['title']}")

# Step 1: 대화창
st.subheader("💬 대화창")
for turn in st.session_state["conversation"]:
    if turn["role"] == "user":
        st.markdown(
            f"<div style='text-align: right; color: #31333F; padding: 10px 5px;'><b>사용자:</b> {turn['message']}</div>",
            unsafe_allow_html=True,
        )
    elif turn["role"] == "bot":
        st.markdown(
            f"<div style='text-align: left; color: #31333F; padding: 10px 5px;'><b>챗봇:</b> {turn['message']}</div>",
            unsafe_allow_html=True,
        )


# Step 2: 입력 및 처리
st.subheader("💬 의견 또는 질문 입력")


# 사용자 입력 처리 함수
def handle_user_input():
    user_message = st.session_state["user_input"]
    if user_message.strip():  # 빈 문자열 체크
        # 사용자 메시지 저장
        st.session_state["conversation"].append({"role": "user", "message": user_message})

        # 챗봇 응답 생성
        with st.spinner("챗봇이 응답을 생성 중입니다..."):
            try:
                # 기사 정보 준비 (모든 질문에서 기사 내용 전달)
                article_info = ""
                article = st.session_state.get("article")
                if article:
                    article_info = f"기사 ID: {article['id']}\n기사 제목: {article['title']}\n기사 내용: {article['text']}"
                
                # 이전 대화 기록을 맥락으로 구성
                conversation_context = ""
                if len(st.session_state["conversation"]) > 1:
                    conversation_context = "이전 대화:\n"
                    for turn in st.session_state["conversation"][:-1]:  # 현재 질문 제외
                        role = "사용자" if turn["role"] == "user" else "AI"
                        conversation_context += f"{role}: {turn['message']}\n"
                
                # 전체 맥락 구성
                full_context = f"{article_info}\n\n{conversation_context}".strip()
                
                chatbot_reply = chatgpt_response(
                    context=full_context,
                    query=f"사용자가 '{user_message}'라고 말했어. 기사 내용과 이전 대화를 참고하여 질문에 대한 응답 또는 다른 의견을 제시해줘. 말이 끊기지 않도록 300 토큰 제한에 신경 쓰며 4줄 안으로 요약해서 답변해줘",
                )
                # 챗봇 응답 저장
                st.session_state["conversation"].append({"role": "bot", "message": chatbot_reply})
            except Exception as e:
                st.error(f"응답 생성 실패: {e}")
        # 입력창 초기화
        st.session_state["user_input"] = ""


st.text_input(
    "✏️ 메시지를 입력하세요:",
    key="user_input",
    on_change=handle_user_input,
)

# Step 3: AI 대화 기반 추천

if st.button("💡 현재 대화를 기반으로 기사 추천받기", help="지금까지의 대화 내용을 분석하여 관련 기사를 추천합니다"):
    if len(st.session_state["conversation"]) > 0:
        with st.spinner("AI가 대화 내용을 분석하여 기사를 추천 중..."):
            try:
                # 챗봇에서 사용한 정보를 재사용하여 Flask로 전달
                article = st.session_state.get("article")
                
                # 기사 정보 구성 (챗봇에서 사용한 것과 동일)
                article_info = ""
                if article:
                    article_info = f"기사 ID: {article['id']}\n기사 제목: {article['title']}\n기사 내용: {article['text']}"
                
                # Flask 서버의 AI 추천 API 호출
                request_data = {
                    "conversation": st.session_state["conversation"],
                    "article_info": article_info
                }
                
                response = requests.post("http://localhost:5002/get_ai_recommendations", 
                                       json=request_data)
                if response.status_code == 200:
                    ai_recommendation = response.json()["recommendation"]
                    st.session_state["ai_recommendation"] = ai_recommendation
                else:
                    st.error("AI 추천 생성 실패")
            except Exception as e:
                st.error(f"AI 추천 요청 실패: {e}")
    else:
        st.warning("대화 기록이 없습니다. 먼저 대화를 시작해주세요.")

# 추천 결과 표시 및 챗봇에 추가
if "ai_recommendation" in st.session_state:
    st.subheader("🤖 AI 대화 기반 추천 결과")
    
    # 추천 결과를 파싱하여 하이퍼링크로 변환
    recommendation_text = st.session_state["ai_recommendation"]
    
    # 하이퍼링크 형식으로 변환
    import re
    
    # 추천 결과를 라인별로 분리하여 처리
    lines = recommendation_text.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:  # 빈 줄은 그대로 유지
            processed_lines.append(line)
            i += 1
            continue
            
        # 숫자로 시작하는 라인 (기사 제목 라인) 처리
        if re.match(r'^\d+\.', line):
            # 제목에서 (ID: X) 부분 제거
            title = re.sub(r'\s*\(ID:\s*\d+\)', '', line)
            title = re.sub(r'^\d+\.\s*', '', title).strip()
            # "제목:" 텍스트 제거
            title = re.sub(r'^제목:\s*', '', title).strip()
            
            # 다음 라인들을 확인하여 URL과 설명 찾기
            url = None
            description = ""
            
            # 다음 몇 라인을 확인
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if 'http://localhost:5002/article/' in next_line:
                    # URL 추출
                    url_match = re.search(r'http://localhost:5002/article/\d+', next_line)
                    if url_match:
                        url = url_match.group(0)
                        
                        # URL 이후의 텍스트 (추천 이유) 추출
                        url_end = url_match.end()
                        remaining_text = next_line[url_end:].strip()
                        if remaining_text.startswith('–') or remaining_text.startswith('-'):
                            remaining_text = remaining_text[1:].strip()
                        description = remaining_text
                    break
            
            # 마크다운 링크 형식으로 변환
            if url:
                if description:
                    processed_lines.append(f"[{title}]({url}) - {description}")
                else:
                    processed_lines.append(f"[{title}]({url})")
                # URL 라인까지 건너뛰기
                i = j + 1
            else:
                processed_lines.append(line)
                i += 1
        else:
            processed_lines.append(line)
            i += 1

    
    recommendation_with_links = '\n'.join(processed_lines)
    
    st.markdown(recommendation_with_links)



# Step 4: 토론 종료
if st.button("🔚 토론 종료"):
    # 모든 상태 초기화
    st.session_state["conversation"] = []
    st.session_state["article"] = None
    if "ai_recommendation" in st.session_state:
        del st.session_state["ai_recommendation"]
    st.session_state["initial_statement_done"] = True
    st.success("토론이 종료되었습니다. 새로운 토론을 시작할 수 있습니다.")
