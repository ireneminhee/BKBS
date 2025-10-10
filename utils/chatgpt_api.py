from openai import OpenAI

# OpenAI API 키 설정 (환경 변수에서 가져오거나 직접 설정)
import os
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

def chatgpt_response(context, query):

    # 프롬프트 설정
    messages = [
        {
            "role": "system",
            "content": f"당신은 사용자와 반대 입장에서 토론을 진행하는 AI입니다. 논리적이고 명확하게 의견을 제시하세요. 만약 사용자가 질문을 한다면, 친절하게 대답해주는 비서가 되세요.\n\n{context}"
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # 챗봇 프롬프트를 터미널에 출력
    print("=" * 80)
    print("🤖 챗봇 프롬프트:")
    print("=" * 80)
    print(f"시스템 메시지: {messages[0]['content']}")
    print("-" * 80)
    print(f"사용자 메시지: {messages[1]['content']}")
    print("=" * 80)
    
    try:
        # ChatGPT API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 사용할 모델
            messages=messages,
            max_tokens=300,  # 출력 토큰 제한
            temperature=0.7  # 창의성 조절
        )
        # 응답 반환
        chatbot_reply = response.choices[0].message.content.strip()
        
        # 챗봇 응답을 터미널에 출력
        print("=" * 80)
        print("🤖 챗봇 응답:")
        print("=" * 80)
        print(chatbot_reply)
        print("=" * 80)
        
        return chatbot_reply
    except Exception as e:
        # 에러 메시지 반환
        error_msg = f"ChatGPT API 호출 실패: {str(e)}"
        print(f"❌ 챗봇 오류: {error_msg}")
        return error_msg
