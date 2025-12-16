#!/bin/bash

# 서버 업데이트
echo "서버 업데이트 중..."
sudo apt update -y && sudo apt upgrade -y

# 필수 패키지 설치
echo "필수 패키지 설치 중..."
sudo apt install -y python3-pip python3-dev python3-venv git

# pip 최신 버전으로 업데이트
echo "pip 최신 버전으로 업데이트 중..."
python3 -m pip install --upgrade pip

# 필요한 Python 라이브러리 설치
echo "필요한 Python 라이브러리 설치 중..."
pip3 install python-telegram-bot pyupbit

# 텔레그램 봇 코드 다운로드
echo "텔레그램 봇 코드 다운로드 중..."
mkdir -p /home/ubuntu/coin_trading_bot
cd /home/ubuntu/coin_trading_bot

# bot.py 파일 생성
cat <<EOL > bot.py
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, filters
import pyupbit
import asyncio

# 텔레그램 봇 토큰 (BotFather에서 발급받은 토큰을 넣으세요)
TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # 'YOUR_TELEGRAM_BOT_TOKEN' 부분을 실제 봇 토큰으로 교체

# 업비트 API 설정
access_key = 'YOUR_ACCESS_KEY'  # 업비트 API에서 발급받은 ACCESS KEY
secret_key = 'YOUR_SECRET_KEY'  # 업비트 API에서 발급받은 SECRET KEY
upbit = pyupbit.Upbit(access_key, secret_key)

# 로그 설정
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# /start 명령어 처리
async def start(update: Update, context):
    """/start 명령어가 입력되었을 때 응답할 함수"""
    await update.message.reply_text("안녕하세요! 저는 당신의 코인 자동화 봇입니다.")

# /help 명령어 처리
async def help(update: Update, context):
    """/help 명령어가 입력되었을 때 응답할 함수"""
    await update.message.reply_text("도움말을 원하시면 언제든지 말씀해주세요.")

# /price 명령어 처리 (현재 비트코인 가격 조회)
async def price(update: Update, context):
    """/price 명령어가 입력되었을 때 응답할 함수"""
    ticker = "KRW-BTC"  # 예시로 비트코인 가격 가져오기
    price = pyupbit.get_current_price(ticker)
    await update.message.reply_text(f"현재 {ticker} 가격은 {price} 원입니다.")

# 메인 함수
async def main():
    """봇 실행"""
    # 텔레그램 봇 애플리케이션 설정
    application = Application.builder().token(TOKEN).build()

    # 명령어 처리 함수들
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("price", price))

    # 봇 시작
    logging.info("봇이 시작되었습니다.")
    await application.run_polling()

# 프로그램 시작
if __name__ == '__main__':
    asyncio.run(main())  # 비동기 실행
EOL

# 업비트 API 코드 파일 생성
cat <<EOL > upbit_api.py
import pyupbit

# 업비트 API 연결
access_key = 'YOUR_ACCESS_KEY'  # 업비트 API에서 발급받은 ACCESS KEY
secret_key = 'YOUR_SECRET_KEY'  # 업비트 API에서 발급받은 SECRET KEY
upbit = pyupbit.Upbit(access_key, secret_key)

# 예시: 현재 비트코인 가격 가져오기
def get_current_price(ticker="KRW-BTC"):
    """현재 코인 가격 가져오기"""
    price = pyupbit.get_current_price(ticker)
    return price

# 예시: 잔고 조회
def get_balance():
    """내 계좌의 잔고 조회"""
    balance = upbit.get_balance("KRW")  # KRW 잔고 조회
    return balance

# 테스트로 가격을 출력해보기
print(get_current_price())
print(get_balance())
EOL

# 봇 실행 테스트
echo "봇을 실행하려면 'python3 bot.py'를 사용하세요."
echo "봇 코드가 /home/ubuntu/coin_trading_bot에 저장되었습니다."

# 실행 권한 부여
chmod +x /home/ubuntu/coin_trading_bot/bot.py

# 실행 방법 안내
echo "스크립트가 완료되었습니다. 봇을 실행하려면 다음 명령어를 입력하세요:"
echo "python3 /home/ubuntu/coin_trading_bot/bot.py"
