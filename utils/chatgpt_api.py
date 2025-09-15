import openai

# OpenAI API 키 설정 (환경 변수에서 가져오거나 직접 설정)
import os
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

def chatgpt_response(context, query):

    # 프롬프트 설정
    messages = [
        {
            "role": "system",
            "content": "당신은 사용자와 반대 입장에서 토론을 진행하는 AI입니다. 논리적이고 명확하게 의견을 제시하세요. 만약 사용자가 질문을 한다면, 친절하게 대답해주는 비서가 되세요."
        },
        {
            "role": "user",
            "content": f"상황: {context}\n발언: {query}"
        }
    ]
    try:
        # ChatGPT API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델
            messages=messages,
            max_tokens=300,  # 출력 토큰 제한
            temperature=0.7  # 창의성 조절
        )
        # 응답 반환
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # 에러 메시지 반환
        return f"ChatGPT API 호출 실패: {str(e)}"