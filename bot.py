import logging
from telegram import Update
from telegram.ext import Application, CommandHandler
import pyupbit
import openai
import asyncio

# 텔레그램 봇 토큰 (BotFather에서 발급받은 토큰을 넣으세요)
TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

# 텔레그램 사용자 ID (봇과 대화한 사용자의 ID)
TELEGRAM_USER_ID = 'YOUR_USER_ID'

# 업비트 API 설정
access_key = 'YOUR_ACCESS_KEY'
secret_key = 'YOUR_SECRET_KEY'
upbit = pyupbit.Upbit(access_key, secret_key)

# OpenAI API 설정
openai.api_key = 'YOUR_OPENAI_API_KEY'

# 로그 설정
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# 시장 조건별 매수 비율
def calculate_investment_strategy(balance, market_condition):
    if market_condition == "BEAR":
        return balance * 0.4  # BEAR 상태에서는 40% 사용
    elif market_condition == "RANGE":
        return balance * 0.7  # RANGE 상태에서는 70% 사용
    elif market_condition == "BULL":
        return balance  # BULL 상태에서는 100% 사용

# 현재 보유 코인 중 GAS, VTHO, APENFT 제외
def filter_usable_coins():
    excluded_coins = ["GAS", "VTHO", "APENFT"]
    balance = upbit.get_balances()
    usable_coins = []

    for coin in balance:
        if coin['currency'] not in excluded_coins:
            usable_coins.append(coin['currency'])

    return usable_coins

# /start 명령어 처리
async def start(update: Update, context):
    await update.message.reply_text("안녕하세요! 저는 당신의 코인 자동화 봇입니다.")

# /help 명령어 처리
async def help(update: Update, context):
    await update.message.reply_text("도움말을 원하시면 언제든지 말씀해주세요.")

# /price 명령어 처리 (현재 비트코인 가격 조회)
async def price(update: Update, context):
    ticker = "KRW-BTC"
    price = pyupbit.get_current_price(ticker)
    await update.message.reply_text(f"현재 {ticker} 가격은 {price} 원입니다.")

# AI를 사용한 매수 신호 예측
def get_trading_signal(text):
    response = openai.Completion.create(
        engine="gpt-5.2",
        prompt=text,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# 예시: GPT 모델을 통한 예측 출력
async def get_signal(update: Update, context):
    signal = get_trading_signal("현재 비트코인 가격 상승을 예상합니다. 매수 시점인가요?")
    await update.message.reply_text(f"GPT 예측: {signal}")

# 매수 실행
def execute_purchase(amount_in_krw):
    ticker = "KRW-BTC"
    price = pyupbit.get_current_price(ticker)
    if price is not None:
        amount_to_buy = amount_in_krw / price
        upbit.buy_market_order(ticker, amount_to_buy)
        return f"{ticker} 구매: {amount_to_buy} BTC, 가격: {price} 원"
    else:
        return "가격을 가져올 수 없습니다."

# /buy 명령어 처리 (매수 실행)
async def buy(update: Update, context):
    balance = upbit.get_balance("KRW")
    market_condition = "BULL"  # 예시로 BULL 상태
    investment_amount = calculate_investment_strategy(balance, market_condition)
    
    # 매수 실행
    result = execute_purchase(investment_amount)
    await update.message.reply_text(result)

# 메인 함수
async def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("price", price))
    application.add_handler(CommandHandler("signal", get_signal))  # 새로운 명령어 처리
    application.add_handler(CommandHandler("buy", buy))  # 매수 명령어 처리

    logging.info("봇이 시작되었습니다.")
    await application.run_polling()

# 프로그램 시작
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
