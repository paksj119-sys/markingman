#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/coinbot"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$APP_DIR"
cd "$APP_DIR"

# 1) venv
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# 2) deps
pip install pyupbit python-dotenv requests tenacity openai

# 3) .env template (수정 필요)
cat > .env <<'ENV'
# ===== Runtime =====
LIVE_TRADING=0                 # 1로 바꾸면 실거래
SCAN_INTERVAL_SEC=300          # 5분
REPORT_EVERY_SEC=10800         # 3시간

# ===== Upbit =====
UPBIT_ACCESS_KEY=PASTE_ME
UPBIT_SECRET_KEY=PASTE_ME

# ===== Telegram =====
TELEGRAM_BOT_TOKEN=PASTE_ME
TELEGRAM_CHAT_ID=PASTE_ME

# ===== OpenAI =====
OPENAI_API_KEY=PASTE_ME

# 모델(요구사항)
OPENAI_MODEL_FAST=gpt-4.1-mini
OPENAI_MODEL_STRONG=gpt-5.2

# BUY 확신 임계치
BUY_MIN_CONF=0.85

# ===== Universe =====
TOP_N=50
EXCLUDE_TICKERS=KRW-GAS,KRW-VTHO,KRW-APENFT

# ===== Risk defaults (트리거 생성용) =====
STOP_LOSS=-0.08
TAKE_PROFIT=0.15
PARTIAL_TP_RATIO=0.5
TRAIL_START=0.10
TRAIL_GAP=0.05
ENV

# 4) code files
cat > config.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

def _as_bool(v: str, default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _as_int(v: str, default: int) -> int:
    try: return int(str(v).strip())
    except Exception: return default

def _as_float(v: str, default: float) -> float:
    try: return float(str(v).strip())
    except Exception: return default

def _as_list(v: str) -> list[str]:
    if not v: return []
    return [x.strip() for x in v.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    LIVE_TRADING: bool
    SCAN_INTERVAL_SEC: int
    REPORT_EVERY_SEC: int

    UPBIT_ACCESS_KEY: str
    UPBIT_SECRET_KEY: str

    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    OPENAI_API_KEY: str
    OPENAI_MODEL_FAST: str
    OPENAI_MODEL_STRONG: str
    BUY_MIN_CONF: float

    TOP_N: int
    EXCLUDE_TICKERS: set[str]

    # sizing (요구사항 고정)
    SIZING: dict

    # risk triggers (요구 수치 반영)
    STOP_LOSS: float
    TAKE_PROFIT: float
    PARTIAL_TP_RATIO: float
    TRAIL_START: float
    TRAIL_GAP: float

def load_settings(env_path: str = ".env") -> Settings:
    load_dotenv(env_path)

    sizing = {
        "BEAR":  {"use": 0.40, "max_ports": 3},
        "RANGE": {"use": 0.70, "max_ports": 5},
        "BULL":  {"use": 1.00, "max_ports": 10},
    }

    return Settings(
        LIVE_TRADING=_as_bool(os.getenv("LIVE_TRADING","0"), False),
        SCAN_INTERVAL_SEC=_as_int(os.getenv("SCAN_INTERVAL_SEC","300"), 300),
        REPORT_EVERY_SEC=_as_int(os.getenv("REPORT_EVERY_SEC","10800"), 10800),

        UPBIT_ACCESS_KEY=os.getenv("UPBIT_ACCESS_KEY","").strip(),
        UPBIT_SECRET_KEY=os.getenv("UPBIT_SECRET_KEY","").strip(),

        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip(),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID","").strip(),

        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY","").strip(),
        OPENAI_MODEL_FAST=os.getenv("OPENAI_MODEL_FAST","gpt-4.1-mini").strip(),
        OPENAI_MODEL_STRONG=os.getenv("OPENAI_MODEL_STRONG","gpt-5.2").strip(),
        BUY_MIN_CONF=_as_float(os.getenv("BUY_MIN_CONF","0.85"), 0.85),

        TOP_N=_as_int(os.getenv("TOP_N","50"), 50),
        EXCLUDE_TICKERS=set(_as_list(os.getenv("EXCLUDE_TICKERS",""))),

        SIZING=sizing,

        STOP_LOSS=_as_float(os.getenv("STOP_LOSS","-0.08"), -0.08),
        TAKE_PROFIT=_as_float(os.getenv("TAKE_PROFIT","0.15"), 0.15),
        PARTIAL_TP_RATIO=_as_float(os.getenv("PARTIAL_TP_RATIO","0.5"), 0.5),
        TRAIL_START=_as_float(os.getenv("TRAIL_START","0.10"), 0.10),
        TRAIL_GAP=_as_float(os.getenv("TRAIL_GAP","0.05"), 0.05),
    )
PY

cat > notifier.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
import time
import requests
from config import load_settings

@dataclass
class NotifyEvent:
    level: str   # INFO / TRADE / ERROR
    title: str
    message: str

class Notifier:
    def __init__(self) -> None:
        self.s = load_settings()
        self._dedup: dict[str, float] = {}

    def notify(self, ev: NotifyEvent, dedup_key: str, dedup_sec: int) -> None:
        now = time.time()
        if dedup_sec > 0:
            last = self._dedup.get(dedup_key, 0.0)
            if now - last < dedup_sec:
                return
            self._dedup[dedup_key] = now

        token = self.s.TELEGRAM_BOT_TOKEN
        chat_id = self.s.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return

        text = f"[coinbot][{ev.level}] {ev.title}\n{ev.message}".strip()
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:3900]},
                timeout=10,
            )
        except Exception:
            # 텔레그램 실패로 봇 전체가 죽으면 안 됨
            return
PY

cat > upbit_client.py <<'PY'
from __future__ import annotations
import time
import os
from typing import Any
import pyupbit
from tenacity import retry, stop_after_attempt, wait_exponential

class UpbitClient:
    def __init__(self, access: str, secret: str) -> None:
        self.upbit = pyupbit.Upbit(access, secret)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=3))
    def get_balances(self) -> list[dict[str, Any]]:
        b = self.upbit.get_balances()
        return b or []

    def get_krw_cash(self) -> float:
        try:
            b = self.get_balances()
            row = next((x for x in b if x.get("currency") == "KRW"), None)
            if not row:
                return 0.0
            return float(row.get("balance") or 0.0) + float(row.get("locked") or 0.0)
        except Exception:
            return 0.0

    def buy_market(self, market: str, krw: float) -> None:
        self.upbit.buy_market_order(market, krw)

    def sell_market(self, market: str, qty: float) -> None:
        self.upbit.sell_market_order(market, qty)
PY

cat > portfolio.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json, os, time

DEFAULT_FILE = "portfolio.json"

@dataclass
class Position:
    ticker: str
    status: str = "OPEN"   # OPEN/CLOSED
    qty: float = 0.0
    entry: float = 0.0
    high: float = 0.0
    partial_done: bool = False
    open_ts: int = 0
    last_ai: str = ""

    def normalize(self) -> "Position":
        self.qty = float(self.qty or 0.0)
        self.entry = float(self.entry or 0.0)
        self.high = float(self.high or 0.0)
        self.open_ts = int(self.open_ts or 0) or int(time.time())
        self.status = self.status or "OPEN"
        return self

class Portfolio:
    def __init__(self, positions: Optional[Dict[str, Position]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        self.positions: Dict[str, Position] = positions or {}
        self.meta: Dict[str, Any] = meta or {}

    @property
    def last_krw_cash_snapshot(self) -> float:
        return float((self.meta or {}).get("last_krw_cash_snapshot") or 0.0)

    @last_krw_cash_snapshot.setter
    def last_krw_cash_snapshot(self, v: float) -> None:
        self.meta["last_krw_cash_snapshot"] = float(v)

    def open_positions(self) -> Dict[str, Position]:
        return {k: v for k, v in self.positions.items() if v.status == "OPEN" and float(v.qty or 0.0) > 0.0}

    def upsert(self, pos: Position) -> None:
        self.positions[pos.ticker] = pos.normalize()

    def close(self, ticker: str) -> None:
        if ticker in self.positions:
            self.positions[ticker].status = "CLOSED"
            self.positions[ticker].qty = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "meta": self.meta,
        }

    def save(self, path: str = DEFAULT_FILE) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def positions_value(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.open_positions().items():
            px = float(prices.get(sym) or 0.0)
            total += float(pos.qty or 0.0) * px
        return float(total)

def load_portfolio(path: str = DEFAULT_FILE) -> Portfolio:
    if not os.path.exists(path):
        return Portfolio()
    try:
        raw = json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return Portfolio()
    positions: Dict[str, Position] = {}
    for k, v in (raw.get("positions") or {}).items():
        try:
            positions[k] = Position(**v).normalize()
        except Exception:
            continue
    meta = raw.get("meta") or {}
    return Portfolio(positions=positions, meta=meta)
PY

cat > universe.py <<'PY'
from __future__ import annotations
from typing import Dict, List, Any
import requests
from config import load_settings

UPBIT_MARKET_ALL = "https://api.upbit.com/v1/market/all"
UPBIT_TICKER = "https://api.upbit.com/v1/ticker"

def _get_markets() -> List[Dict[str, Any]]:
    r = requests.get(UPBIT_MARKET_ALL, params={"isDetails": "true"}, timeout=10)
    r.raise_for_status()
    return r.json() or []

def _get_tickers(markets: List[str]) -> List[Dict[str, Any]]:
    # upbit ticker endpoint supports multiple markets
    out: List[Dict[str, Any]] = []
    for i in range(0, len(markets), 100):
        chunk = markets[i:i+100]
        r = requests.get(UPBIT_TICKER, params={"markets": ",".join(chunk)}, timeout=10)
        r.raise_for_status()
        out.extend(r.json() or [])
    return out

def build_universe(top_n: int | None = None) -> List[Dict[str, Any]]:
    s = load_settings()
    top_n = int(top_n or s.TOP_N)

    markets = _get_markets()
    krw = []
    for m in markets:
        mk = str(m.get("market") or "")
        if not mk.startswith("KRW-"):
            continue
        # market_warning=CAUTION 제외(요구사항)
        if str(m.get("market_warning") or "").upper() == "CAUTION":
            continue
        if mk in s.EXCLUDE_TICKERS:
            continue
        krw.append(mk)

    # 24h 거래대금 기준 상위
    ticks = _get_tickers(krw)
    # acc_trade_price_24h is not always present; use acc_trade_price (which is 24h)
    ticks.sort(key=lambda x: float(x.get("acc_trade_price") or 0.0), reverse=True)

    out = []
    for x in ticks[:top_n]:
        out.append({
            "ticker": x.get("market"),
            "price": float(x.get("trade_price") or 0.0),
            "volume": float(x.get("acc_trade_price") or 0.0),
            "signed_change_rate": float(x.get("signed_change_rate") or 0.0),
        })
    return out
PY

cat > market.py <<'PY'
from __future__ import annotations
import pyupbit

def _ema(series, span: int) -> float:
    # simple EMA for last value
    alpha = 2.0 / (span + 1.0)
    ema = series[0]
    for v in series[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)

def market_state() -> str:
    """
    BTC 기반 단순 판정:
    - BULL: price > EMA50 > EMA200 (1H)
    - BEAR: price < EMA50 < EMA200 (1H)
    - else RANGE
    """
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=220)
        closes = [float(x) for x in df["close"].tolist() if x is not None]
        if len(closes) < 210:
            return "RANGE"
        price = closes[-1]
        ema50 = _ema(closes[-200:], 50)
        ema200 = _ema(closes[-210:], 200)
        if price > ema50 > ema200:
            return "BULL"
        if price < ema50 < ema200:
            return "BEAR"
        return "RANGE"
    except Exception:
        return "RANGE"

def range_to_bull_ramp_use() -> float:
    """
    RANGE에서 상승 전환 강도에 따라 0.30/0.50/0.70
    기준: BTC 최근 30분(5분봉 6개) 상승률
    - >= 5% : 0.70
    - >= 3% : 0.50
    - >= 1.5% : 0.30
    - else 0.0
    """
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=7)
        closes = [float(x) for x in df["close"].tolist() if x is not None]
        if len(closes) < 7:
            return 0.0
        base = closes[-7]
        last = closes[-1]
        if base <= 0:
            return 0.0
        ret = last / base - 1.0
        if ret >= 0.05:
            return 0.70
        if ret >= 0.03:
            return 0.50
        if ret >= 0.015:
            return 0.30
        return 0.0
    except Exception:
        return 0.0
PY

cat > risk.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from portfolio import Portfolio, Position
from config import load_settings

@dataclass
class RiskDecision:
    action: str   # HOLD / PARTIAL_TP / CLOSE
    reason: str
    emergency: bool = False

def update_high(pos: Position, price: float) -> None:
    if price > float(pos.high or 0.0):
        pos.high = float(price)

def rule_based_risk(pos: Position, price: float, stop_loss: float, take_profit: float,
                    partial_tp_ratio: float, trail_start: float, trail_gap: float) -> RiskDecision:
    """
    트리거만 생성. 실제 매도는 AI 승인 필요(옵션B).
    """
    entry = float(pos.entry or 0.0)
    if entry <= 0:
        return RiskDecision("HOLD", "no_entry")

    ret = float(price) / entry - 1.0

    # STOP LOSS trigger
    if ret <= float(stop_loss):
        return RiskDecision("CLOSE", f"STOP_LOSS hit ret={ret:.2%}", emergency=True)

    # TAKE PROFIT trigger (partial once)
    if (not pos.partial_done) and ret >= float(take_profit):
        return RiskDecision("PARTIAL_TP", f"TAKE_PROFIT ret={ret:.2%} (ratio={partial_tp_ratio:.0%})", emergency=False)

    # TRAILING trigger (10% up reached, then -5% from high)
    if float(pos.high or 0.0) > 0 and ret >= float(trail_start):
        dd = float(price) / float(pos.high) - 1.0
        if dd <= -abs(float(trail_gap)):
            return RiskDecision("CLOSE", f"TRAIL stop dd={dd:.2%} (high={pos.high:.0f})", emergency=False)

    return RiskDecision("HOLD", f"ok ret={ret:.2%}")

def apply_rule_risk_to_portfolio(pf: Portfolio, prices: Dict[str, float]) -> List[Tuple[str, RiskDecision]]:
    s = load_settings()
    out: List[Tuple[str, RiskDecision]] = []
    for sym, pos in pf.open_positions().items():
        px = float(prices.get(sym) or 0.0)
        if px <= 0:
            continue
        update_high(pos, px)
        d = rule_based_risk(
            pos, px,
            stop_loss=s.STOP_LOSS,
            take_profit=s.TAKE_PROFIT,
            partial_tp_ratio=s.PARTIAL_TP_RATIO,
            trail_start=s.TRAIL_START,
            trail_gap=s.TRAIL_GAP,
        )
        out.append((sym, d))
    return out
PY

cat > ai_engine.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from config import load_settings

# OpenAI SDK (best effort)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@dataclass
class BuyAIResult:
    ok: bool
    conf: float
    reason: str

@dataclass
class RiskAIResult:
    action: str   # HOLD / PARTIAL_TP / CLOSE
    conf: float
    reason: str

def _json_sanitize(s: str) -> str:
    s = s.strip()
    # remove code fences if present
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s else s
    return s.strip()

class AIEngine:
    def __init__(self) -> None:
        self.s = load_settings()
        self.client = OpenAI(api_key=self.s.OPENAI_API_KEY) if OpenAI else None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.6, min=0.6, max=3))
    def _call_json(self, model: str, system: str, user: str) -> Dict[str, Any]:
        if not self.client:
            # no SDK: fail closed
            return {"ok": False, "conf": 0.0, "reason": "openai_sdk_missing"}

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(_json_sanitize(content))
        except Exception:
            return {"ok": False, "conf": 0.0, "reason": "json_parse_fail", "raw": content[:400]}

    def decide_buy(self, market_state: str, coin: Dict[str, Any]) -> BuyAIResult:
        """
        요구사항: '확신' 있는 코인만 매수.
        두 모델(FAST/STRONG) 모두 ok AND conf>=BUY_MIN_CONF 이어야 BUY.
        """
        sys = (
            "You are a crypto trading decision engine.\n"
            "Return ONLY JSON: {\"ok\":true/false,\"conf\":0..1,\"reason\":\"...\"}\n"
            "Be conservative: approve ONLY if you are confident the coin will go up soon.\n"
        )
        user = json.dumps({
            "task": "BUY_DECISION",
            "market_state": market_state,
            "coin": coin,
            "rules": {
                "approve_only_if_confident": True,
                "min_conf": self.s.BUY_MIN_CONF,
            }
        }, ensure_ascii=False)

        r1 = self._call_json(self.s.OPENAI_MODEL_FAST, sys, user)
        r2 = self._call_json(self.s.OPENAI_MODEL_STRONG, sys, user)

        ok1 = bool(r1.get("ok")) and float(r1.get("conf") or 0.0) >= self.s.BUY_MIN_CONF
        ok2 = bool(r2.get("ok")) and float(r2.get("conf") or 0.0) >= self.s.BUY_MIN_CONF

        conf = min(float(r1.get("conf") or 0.0), float(r2.get("conf") or 0.0))
        reason = f"fast={r1.get('reason')} | strong={r2.get('reason')}"
        return BuyAIResult(ok=bool(ok1 and ok2), conf=conf, reason=reason[:1200])

    def decide_risk_for_position(self, market_state: str, pos: Dict[str, Any], price: float,
                                 trigger: str, trigger_reason: str) -> RiskAIResult:
        """
        옵션B: 매도는 AI가 승인해야만 실행. (트리거=STOP/TP/TRAIL은 참고 정보)
        Return JSON: { "action": "HOLD|PARTIAL_TP|CLOSE", "conf":0..1, "reason":"..." }
        """
        sys = (
            "You are a crypto risk manager.\n"
            "Return ONLY JSON: {\"action\":\"HOLD|PARTIAL_TP|CLOSE\",\"conf\":0..1,\"reason\":\"...\"}\n"
            "If unsure, choose HOLD.\n"
        )
        user = json.dumps({
            "task": "RISK_DECISION",
            "market_state": market_state,
            "position": pos,
            "price": price,
            "trigger": trigger,
            "trigger_reason": trigger_reason,
            "policy": {
                "ai_veto_all_sells": True
            }
        }, ensure_ascii=False)

        r1 = self._call_json(self.s.OPENAI_MODEL_FAST, sys, user)
        r2 = self._call_json(self.s.OPENAI_MODEL_STRONG, sys, user)

        # Conservative merge: if models disagree, HOLD
        a1 = str(r1.get("action") or "HOLD").upper()
        a2 = str(r2.get("action") or "HOLD").upper()
        if a1 not in ("HOLD","PARTIAL_TP","CLOSE"): a1 = "HOLD"
        if a2 not in ("HOLD","PARTIAL_TP","CLOSE"): a2 = "HOLD"

        if a1 != a2:
            action = "HOLD"
        else:
            action = a1

        conf = min(float(r1.get("conf") or 0.0), float(r2.get("conf") or 0.0))
        reason = f"fast={r1.get('reason')} | strong={r2.get('reason')} | trigger={trigger}:{trigger_reason}"
        return RiskAIResult(action=action, conf=conf, reason=reason[:1400])
PY

cat > trader.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import time
import pyupbit

from config import load_settings
from portfolio import Portfolio, Position
from risk import apply_rule_risk_to_portfolio
from ai_engine import AIEngine
from market import range_to_bull_ramp_use

MIN_ORDER_KRW = 5000.0  # 업비트 KRW 최소주문(옵션B: 포트 자동조절 없음, 미만이면 스킵)

@dataclass
class TradeAction:
    kind: str   # BUY / SELL_PARTIAL / SELL_ALL
    ticker: str
    krw: float = 0.0
    qty: float = 0.0
    reason: str = ""
    emergency: bool = False

def safe_prices(tickers: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not tickers:
        return out
    # pyupbit can accept list sometimes; use per ticker for stability
    for t in tickers:
        try:
            px = pyupbit.get_current_price(t)
            if px:
                out[t] = float(px)
        except Exception:
            continue
    return out

def build_actions(state: str, pf: Portfolio, universe: List[Dict[str, Any]], prices: Dict[str, float], krw_total: float) -> List[TradeAction]:
    s = load_settings()
    ai = AIEngine()
    actions: List[TradeAction] = []

    # 1) Risk decisions for open positions
    risk_pairs = apply_rule_risk_to_portfolio(pf, prices)
    for sym, rd in risk_pairs:
        if rd.action == "HOLD":
            continue
        pos = pf.open_positions().get(sym)
        if not pos:
            continue
        px = float(prices.get(sym) or 0.0)
        if px <= 0:
            continue

        ai_r = ai.decide_risk_for_position(
            market_state=state,
            pos={
                "ticker": pos.ticker, "qty": pos.qty, "entry": pos.entry, "high": pos.high,
                "partial_done": pos.partial_done, "open_ts": pos.open_ts
            },
            price=px,
            trigger=rd.action,
            trigger_reason=rd.reason,
        )

        if ai_r.action == "PARTIAL_TP":
            qty = float(pos.qty) * float(s.PARTIAL_TP_RATIO)
            if qty > 0:
                actions.append(TradeAction(kind="SELL_PARTIAL", ticker=sym, qty=qty, reason=f"AI approve PARTIAL | {ai_r.reason}", emergency=False))
                pos.partial_done = True

        elif ai_r.action == "CLOSE":
            qty = float(pos.qty)
            if qty > 0:
                actions.append(TradeAction(kind="SELL_ALL", ticker=sym, qty=qty, reason=f"AI approve CLOSE | {ai_r.reason}", emergency=bool(rd.emergency)))

    # 2) Buy sizing (요구사항)
    use = float(s.SIZING[state]["use"])
    max_ports = int(s.SIZING[state]["max_ports"])

    # RANGE -> BULL ramp (30/50/70) only when state=RANGE
    if state == "RANGE":
        ramp = float(range_to_bull_ramp_use() or 0.0)
        if ramp > 0:
            use = ramp  # 0.30/0.50/0.70

    invest = float(krw_total) * float(use)

    # 이미 열려있는 포지션 수
    open_syms = set(pf.open_positions().keys())
    if len(open_syms) >= max_ports:
        return actions

    per_port = invest / float(max_ports) if max_ports > 0 else 0.0
    if per_port < MIN_ORDER_KRW:
        # 옵션B: 자동 조절 없음. 최소주문 미달이면 BUY 전부 스킵
        return actions

    # buy candidates: universe 순서대로, 미보유만, 최대 슬롯만큼
    slots = max_ports - len(open_syms)
    candidates = [c for c in universe if c["ticker"] not in open_syms]
    candidates = candidates[: max(0, slots)]

    for coin in candidates:
        sym = coin["ticker"]
        # AI 확신 매수
        br = ai.decide_buy(state, coin)
        if not br.ok:
            continue
        actions.append(TradeAction(kind="BUY", ticker=sym, krw=float(per_port), reason=f"AI BUY conf={br.conf:.2f} | {br.reason}", emergency=False))
        # 포트 선점(중복 BUY 방지)
        open_syms.add(sym)
        if len(open_syms) >= max_ports:
            break

    return actions

def execute_actions(pf: Portfolio, actions: List[TradeAction], prices: Dict[str, float]) -> List[str]:
    """
    LIVE_TRADING=False: 시뮬레이션 로그만.
    LIVE_TRADING=True : 실주문.
    """
    s = load_settings()
    logs: List[str] = []
    if not actions:
        pf.save()
        return logs

    upbit = None
    if s.LIVE_TRADING:
        try:
            upbit = pyupbit.Upbit(s.UPBIT_ACCESS_KEY, s.UPBIT_SECRET_KEY)
        except Exception:
            upbit = None
            logs.append("ERROR: upbit client init failed")

    for a in actions:
        if a.kind == "BUY":
            if not s.LIVE_TRADING:
                logs.append(f"SIM BUY {a.ticker} krw={a.krw:.0f} | {a.reason}")
                continue
            if upbit is None:
                logs.append(f"ERROR BUY {a.ticker}: no upbit client")
                continue
            try:
                upbit.buy_market_order(a.ticker, a.krw)
                logs.append(f"BUY {a.ticker} krw={a.krw:.0f}")
                # 포지션 반영(대략치)
                px = float(prices.get(a.ticker) or 0.0)
                if px > 0:
                    qty = float(a.krw) / px
                    pf.upsert(Position(ticker=a.ticker, qty=qty, entry=px, high=px, partial_done=False, last_ai="buy"))
            except Exception as e:
                logs.append(f"ERROR BUY {a.ticker}: {e}")

        elif a.kind == "SELL_PARTIAL":
            if not s.LIVE_TRADING:
                logs.append(f"SIM PARTIAL {a.ticker} qty={a.qty:.8f} | {a.reason}")
                continue
            if upbit is None:
                logs.append(f"ERROR SELL_PARTIAL {a.ticker}: no upbit client")
                continue
            try:
                upbit.sell_market_order(a.ticker, a.qty)
                logs.append(f"SELL_PARTIAL {a.ticker} qty={a.qty:.8f}")
                # pf qty 업데이트(대략)
                pos = pf.positions.get(a.ticker)
                if pos:
                    pos.qty = max(0.0, float(pos.qty) - float(a.qty))
            except Exception as e:
                logs.append(f"ERROR SELL_PARTIAL {a.ticker}: {e}")

        elif a.kind == "SELL_ALL":
            if not s.LIVE_TRADING:
                logs.append(f"SIM CLOSE {a.ticker} qty={a.qty:.8f} | {a.reason}")
                continue
            if upbit is None:
                logs.append(f"ERROR SELL_ALL {a.ticker}: no upbit client")
                continue
            try:
                upbit.sell_market_order(a.ticker, a.qty)
                logs.append(f"SELL_ALL {a.ticker} qty={a.qty:.8f}")
                pf.close(a.ticker)
            except Exception as e:
                logs.append(f"ERROR SELL_ALL {a.ticker}: {e}")

    pf.save()
    return logs
PY

cat > 01_portfolio_sync.py <<'PY'
from __future__ import annotations
import os
from dotenv import load_dotenv
import pyupbit
from portfolio import load_portfolio, Position

def main() -> None:
    load_dotenv(".env")
    access = os.getenv("UPBIT_ACCESS_KEY","")
    secret = os.getenv("UPBIT_SECRET_KEY","")
    if not access or not secret:
        print("Missing UPBIT keys in .env")
        return

    u = pyupbit.Upbit(access, secret)
    balances = u.get_balances() or []
    pf = load_portfolio()

    synced = []
    for b in balances:
        cur = str(b.get("currency") or "")
        if cur == "KRW":
            continue
        if cur.strip() == "":
            continue
        market = f"KRW-{cur}"
        try:
            qty = float(b.get("balance") or 0.0) + float(b.get("locked") or 0.0)
            if qty <= 0:
                continue
            avg = float(b.get("avg_buy_price") or 0.0)
            pos = Position(ticker=market, qty=qty, entry=avg, high=avg, partial_done=False, last_ai="sync")
            pf.upsert(pos)
            synced.append(market)
        except Exception:
            continue

    pf.save()
    print("Synced positions:", synced)

if __name__ == "__main__":
    main()
PY

cat > bot.py <<'PY'
from __future__ import annotations
import time
import traceback

from config import load_settings
from notifier import Notifier, NotifyEvent
from upbit_client import UpbitClient
from market import market_state
from universe import build_universe
from portfolio import load_portfolio
from trader import safe_prices, build_actions, execute_actions

ERROR_DEDUP_SEC = 60

def run_once(n: Notifier) -> None:
    s = load_settings()
    pf = load_portfolio()

    state = market_state()
    universe = build_universe(top_n=s.TOP_N)

    tickers = [x["ticker"] for x in universe] + list(pf.open_positions().keys())
    tickers = list(dict.fromkeys(tickers))
    prices = safe_prices(tickers)

    # 총자산 = KRW 현금 + 포지션 평가액 (실거래일 때는 Upbit KRW 실조회)
    krw_cash = 0.0
    if s.LIVE_TRADING:
        uc = UpbitClient(s.UPBIT_ACCESS_KEY, s.UPBIT_SECRET_KEY)
        krw_cash = uc.get_krw_cash()
    else:
        # 시뮬레이션에서는 meta에 쌓아둘 수도 있음(필요 시)
        krw_cash = float(pf.meta.get("krw_cash") or 0.0)

    pos_val = pf.positions_value(prices)
    krw_total = float(krw_cash) + float(pos_val)

    # 입금 감지: "KRW 현금" 기준 (가격변동과 분리)
    prev_cash = float(pf.last_krw_cash_snapshot or 0.0)
    delta = float(krw_cash) - prev_cash
    if prev_cash > 0 and delta > 1000:
        msg = f"deposit_detected +{int(delta)} KRW (prev_cash={int(prev_cash)} now_cash={int(krw_cash)})"
        print("[coinbot]", msg)
        n.notify(NotifyEvent("INFO", "DEPOSIT", msg), dedup_key=f"deposit:{int(time.time())}", dedup_sec=0)

    pf.last_krw_cash_snapshot = float(krw_cash)
    pf.save()

    actions = build_actions(state, pf, universe, prices, krw_total=krw_total)
    logs = execute_actions(pf, actions, prices)

    # journal always
    print(f"[coinbot] cycle state={state} total={int(krw_total)} cash={int(krw_cash)} pos={int(pos_val)} actions={len(actions)}")

    # TRADE 즉시(실거래만)
    if s.LIVE_TRADING:
        trade_lines = [x for x in logs if x and not x.startswith("SIM ")]
        if trade_lines:
            n.notify(NotifyEvent("TRADE", "TRADE", "\n".join(trade_lines)[:3000]),
                     dedup_key=f"trade:{int(time.time())}", dedup_sec=0)

def main() -> None:
    s = load_settings()
    n = Notifier()

    # START 1회(1시간 dedup)
    n.notify(
        NotifyEvent("INFO", "START", f"LIVE_TRADING={s.LIVE_TRADING}, REPORT={int(s.REPORT_EVERY_SEC/3600)}H, SCAN={int(s.SCAN_INTERVAL_SEC/60)}M"),
        dedup_key="start", dedup_sec=3600
    )

    last_report = 0.0
    while True:
        try:
            run_once(n)

            # REPORT 3시간마다
            now = time.time()
            if now - last_report >= float(s.REPORT_EVERY_SEC):
                n.notify(NotifyEvent("INFO", "REPORT", "alive"), dedup_key="report", dedup_sec=int(s.REPORT_EVERY_SEC))
                last_report = now

        except Exception as e:
            msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2500]}"
            print("[coinbot] ERROR", msg)
            n.notify(NotifyEvent("ERROR", "ERROR", msg), dedup_key="err", dedup_sec=ERROR_DEDUP_SEC)

        time.sleep(float(s.SCAN_INTERVAL_SEC))

if __name__ == "__main__":
    main()
PY

# 5) systemd service
SERVICE_PATH="/etc/systemd/system/coinbot.service"
sudo bash -c "cat > '$SERVICE_PATH' <<'UNIT'
[Unit]
Description=Coin Trading Bot (Upbit + AI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/bot.py
Restart=always
RestartSec=5

# journald
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT"

sudo systemctl daemon-reload
sudo systemctl enable coinbot

echo "DONE."
echo
echo "Next:"
echo "1) edit $APP_DIR/.env (keys + LIVE_TRADING=1)"
echo "2) run:  cd $APP_DIR && source .venv/bin/activate && python 01_portfolio_sync.py"
echo "3) start: sudo systemctl restart coinbot && sudo systemctl status coinbot --no-pager"
