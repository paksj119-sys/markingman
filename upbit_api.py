import pyupbit

# 업비트 API 키
access_key = 'YOUR_ACCESS_KEY'  # 업비트 API에서 발급받은 ACCESS KEY
secret_key = 'YOUR_SECRET_KEY'  # 업비트 API에서 발급받은 SECRET KEY

# 업비트 객체 생성
upbit = pyupbit.Upbit(access_key, secret_key)

# 현재 가격 조회 함수
def get_current_price(ticker="KRW-BTC"):
    """현재 코인 가격 가져오기"""
    price = pyupbit.get_current_price(ticker)
    if price is None:
        return "가격을 가져올 수 없습니다."
    return price

# 잔고 조회 함수
def get_balance(currency="KRW"):
    """특정 통화의 잔고 조회 (KRW 또는 BTC 등)"""
    balance = upbit.get_balance(currency)
    return balance

# 특정 코인 매수 함수
def buy_coin(ticker="KRW-BTC", amount_in_krw=10000):
    """특정 코인 매수 함수 (시장가 매수)"""
    price = get_current_price(ticker)
    if price == "가격을 가져올 수 없습니다.":
        return "가격 조회 실패"
    
    amount_to_buy = amount_in_krw / price  # 매수할 코인 수
    order = upbit.buy_market_order(ticker, amount_to_buy)  # 시장가 매수
    return order

# 특정 코인 매도 함수
def sell_coin(ticker="KRW-BTC", amount_in_krw=10000):
    """특정 코인 매도 함수 (시장가 매도)"""
    price = get_current_price(ticker)
    if price == "가격을 가져올 수 없습니다.":
        return "가격 조회 실패"
    
    amount_to_sell = get_balance(ticker.split("-")[1])  # 현재 보유한 코인의 수
    order = upbit.sell_market_order(ticker, amount_to_sell)  # 시장가 매도
    return order

# 시장 상태와 매수 포트 수에 따른 매수 금액 계산 함수
def calculate_investment_by_ports(balance_in_krw, market_condition, max_ports):
    """시장 상태에 맞는 매수 금액을 계산하고, 포트 수에 맞게 나누기"""
    if market_condition == "BEAR":
        total_investment = balance_in_krw * 0.4  # BEAR 상태에서는 40% 사용
    elif market_condition == "RANGE":
        total_investment = balance_in_krw * 0.7  # RANGE 상태에서는 70% 사용
    elif market_condition == "BULL":
        total_investment = balance_in_krw  # BULL 상태에서는 100% 사용
    else:
        total_investment = 0  # 예외 처리

    # 포트 수에 맞게 금액 나누기
    investment_per_port = total_investment / max_ports
    return total_investment, investment_per_port

# 전체 매매 진행 함수 (매수/매도 결정)
def trade_market_condition(balance_in_krw, market_condition, max_ports):
    """매수 및 매도 전략 결정 (BULL, RANGE, BEAR 상태에 따라)"""
    total_investment, investment_per_port = calculate_investment_by_ports(balance_in_krw, market_condition, max_ports)

    # 매수 실행
    if total_investment > 0:
        # 포트 수에 맞게 매수 금액을 분배하여 매수 실행
        result = ""
        for i in range(max_ports):
            ticker = "KRW-BTC"  # 매수할 코인
            result += buy_coin(ticker, investment_per_port) + "\n"
        return f"매수 완료: 총 {total_investment} 원, 포트 수: {max_ports}개, 각 포트당 {investment_per_port} 원"
    
    return "매수할 금액이 없습니다."

# 일정 주기로 매매 진행 함수
def schedule_trading():
    """주기적으로 매매를 진행하는 함수"""
    balance_in_krw = get_balance("KRW")  # KRW 잔고 조회
    if balance_in_krw < 5000:
        return "잔고가 부족하여 매매를 진행할 수 없습니다."
    
    market_condition = "BULL"  # 예시로 BULL 상태 (상승장)
    max_ports = 10  # 최대 10개 포트까지 열 수 있음 (상승장에서는 최대 포트 수 사용)

    # 매매 실행
    result = trade_market_condition(balance_in_krw, market_condition, max_ports)
    return result

# 테스트: 현재 비트코인 가격과 잔고 조회
if __name__ == '__main__':
    # 예시: 매매 실행
    result = schedule_trading()
    print(result)

    # 비트코인 가격 및 잔고 확인
    print(f"현재 비트코인 가격: {get_current_price('KRW-BTC')}")
    print(f"현재 잔고: {get_balance('KRW')} KRW")
