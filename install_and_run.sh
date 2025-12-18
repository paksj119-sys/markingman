#!/bin/bash

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 3 및 pip, 가상환경 설치
sudo apt install python3 python3-pip python3-venv -y

# 가상환경 생성
python3 -m venv trading_bot_env
source trading_bot_env/bin/activate

# 필요한 라이브러리 설치
pip install pyupbit pandas requests

# 봇 코드 생성
echo "
import pyupbit
import pandas as pd
import time

# Upbit API 연결
access_key = 'YOUR_ACCESS_KEY'  # 실제 API 키로 교체
secret_key = 'YOUR_SECRET_KEY'  # 실제 비밀 키로 교체
upbit = pyupbit.Upbit(access_key, secret_key)

# RSI 계산 함수
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# 매수/매도 전략
def should_buy(rsi):
    return rsi[-1] < 30  # RSI가 30 이하일 때 매수

def should_sell(rsi):
    return rsi[-1] > 70  # RSI가 70 이상일 때 매도

# 거래 실행 함수
def execute_trade():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=200)
    rsi = calculate_rsi(df)

    if should_buy(rsi):
        # 매수
        upbit.buy_market_order("KRW-BTC", 10000)  # 예시: 10,000원 만큼 매수
        print("Buying BTC!")
    elif should_sell(rsi):
        # 매도
        upbit.sell_market_order("KRW-BTC", 10000)  # 예시: 10,000원 만큼 매도
        print("Selling BTC!")

# 5분마다 매매 실행
while True:
    execute_trade()
    time.sleep(300)  # 5분마다 실행
" > /root/your-bot.py

# 봇 실행
nohup python3 /root/your-bot.py &
