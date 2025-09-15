#!/bin/bash

# OpenAI API 키 설정 스크립트
echo "=== OpenAI API 키 설정 ==="
echo ""
echo "사용 방법:"
echo "1. OpenAI 웹사이트(https://platform.openai.com/account/api-keys)에서 API 키 생성"
echo "2. 이 스크립트를 실행하고 API 키를 입력하세요"
echo ""

read -p "OpenAI API 키를 입력하세요: " api_key

if [ -n "$api_key" ]; then
    # .env 파일 생성
    echo "OPENAI_API_KEY=$api_key" > .env
    echo "✅ API 키가 .env 파일에 저장되었습니다!"
    echo ""
    echo "⚠️  보안 주의사항:"
    echo "- .env 파일은 .gitignore에 포함되어 Git에 업로드되지 않습니다"
    echo "- API 키를 다른 사람과 공유하지 마세요"
    echo ""
    echo "🚀 이제 애플리케이션을 실행할 수 있습니다:"
    echo "source myenv/bin/activate"
    echo "python main.py"
else
    echo "❌ API 키가 입력되지 않았습니다."
fi
