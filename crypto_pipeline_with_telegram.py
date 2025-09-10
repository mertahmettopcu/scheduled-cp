#!/usr/bin/env python3
"""
Hourly -> daily(noon Istanbul) -> WMA(50/200) -> classify -> upsert (incremental) -> Telegram alerts

Environment:
  SUPABASE_URL, SUPABASE_KEY
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
from supabase import create_client, Client

# -------------------- Config --------------------
DAYS_BACK = 260            # history to fetch (daily points)
NOON_TZ = "Europe/Istanbul"
BINANCE_API = "https://api.binance.com"
SAVE_LAST_N_DAYS = 220     # only upsert most-recent N daily rows / coin
REQUEST_TIMEOUT = 20       # seconds

# Must-have bases (uppercased); we'll map to *USDT* spot pairs if available
MUST_HAVE_BASES = [
    "BTC","ETH","XRP","BNB","SOL","DOGE","TRX","ADA","HYPE","XLM","SUI",
    "LINK","BCH","HBAR","AVAX","TON","LTC","SHIB","DOT","UNI"
]

# -------------------- Env / clients --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log(msg: str) -> None:
    print(msg, flush=True)

# -------------------- Telegram --------------------
def send_telegram(text: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        log("ℹ️ Telegram not configured; skipping alert.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        ok = r.status_code == 200
        if not ok:
            log(f"❌ Telegram error {r.status_code}: {r.text[:200]}")
        return ok
    except Exception as e:
        log(f"❌ Telegram exception: {e}")
        return False

# -------------------- Binance helpers --------------------
def get_binance_spot_symbols() -> set:
    r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    info = r.json()
    return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

def select_pairs(must_have: List[str], take_total: int = 30) -> List[str]:
    listed = get_binance_spot_symbols()
    pairs = []

    skipped = []
    for b in must_have:
        if b.upper() == "USDT":
            continue
        sym = f"{b.upper()}USDT"
        if sym in listed:
            pairs.append(sym)
        else:
            skipped.append(b)

    # fill the rest from top market cap (CoinGecko), but keep must-haves first
    if len(pairs) < take_total:
        try:
            cg = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 50, "page": 1},
                timeout=REQUEST_TIMEOUT,
            )
            cg.raise_for_status()
            symbols = [row["symbol"].upper() + "USDT" for row in cg.json()]
            for sym in symbols:
                if sym.endswith("USDT") and sym in listed and sym not in pairs:
                    pairs.append(sym)
                    if len(pairs) >= take_total:
                        break
        except Exception as e:
            log(f"ℹ️ CoinGecko fallback failed: {e}")

    if skipped:
        log(f"Must-have symbols NOT available on Binance (skipped): {skipped}")
    log(f"✅ Tradable pairs (up to {take_total}, must-haves first): {pairs}")
    return pairs[:take_total]

def fetch_hourly(pair: str, start_ms: int, end_ms: int) -> List[List]:
    out: List[List] = []
    curr = start_ms
    backoff = 0.25

    while curr < end_ms:
        hours = int((end_ms - curr) / (3600 * 1000)) + 1
        limit = max(1, min(1000, hours))
        params = {"symbol": pair, "interval": "1h", "startTime": curr, "endTime": end_ms, "limit": limit}
        try:
            r = requests.get(f"{BINANCE_API}/api/v3/klines", params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            out.extend(data)
            last_open = data[-1][0]
            if last_open <= curr:
                break
            curr = last_open + 3600 * 1000
            time.sleep(0.12)
        except Exception as e:
            log(f"❌ fetch_hourly {pair}: {e}")
            break
    return out

# -------------------- Transform --------------------
def hourly_to_noon_istanbul(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    # Ensure tz-aware UTC
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    # Convert to local tz and pick hour closest to 12:00
    local = df["open_time"].dt.tz_convert(NOON_TZ)
    df["local_date"] = local.dt.date
    df["h_diff"] = (local.dt.hour - 12).abs()
    idx = df.groupby("local_date")["h_diff"].idxmin()
    d = df.loc[idx, ["open_time", "close", "local_date"]].copy()
    d["date"] = pd.to_datetime(d["local_date"])
    d = d.sort_values("date").reset_index(drop=True)
    return d[["date", "close"]]

def wma(series: pd.Series, window: int) -> float:
    s = series.astype(float)
    if len(s) < window or s.isna().any():
        return np.nan
    weights = np.arange(1, window + 1, dtype=float)
    return float((s.to_numpy() * weights).sum() / weights.sum())

def classify(close: float, w50: float, w200: float) -> str:
    if any(pd.isna([close, w50, w200])):
        return "Not enough data"
    lo, hi = min(w50, w200), max(w50, w200)
    if close > hi:
        return "Above both"
    if close < lo:
        return "Below both"
    return "Between"

# -------------------- Main per-coin --------------------
def build_daily_for_symbol(base: str, days_back: int) -> Optional[pd.DataFrame]:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int((days_back + 14) * 24 * 3600 * 1000)  # buffer

    pair = f"{base}USDT"
    raw = fetch_hourly(pair, start_ms, end_ms)
    if not raw:
        return None

    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    daily = hourly_to_noon_istanbul(df)
    if daily.empty:
        return None

    daily = daily.sort_values("date").reset_index(drop=True)
    # WMAs (require full windows)
    daily["wma_50"]  = daily["close"].rolling(window=50, min_periods=50).apply(lambda s: wma(s, 50), raw=False)
    daily["wma_200"] = daily["close"].rolling(window=200, min_periods=200).apply(lambda s: wma(s, 200), raw=False)
    daily["position"] = daily.apply(lambda r: classify(r["close"], r["wma_50"], r["wma_200"]), axis=1)
    daily["previous_position"] = daily["position"].shift(1)

    # keep last N days for upsert
    if SAVE_LAST_N_DAYS:
        daily = daily.tail(SAVE_LAST_N_DAYS).reset_index(drop=True)

    daily["coin"] = base
    daily["date"] = pd.to_datetime(daily["date"]).dt.date.astype(str)  # ISO yyyy-mm-dd strings
    return daily[["coin","date","close","wma_50","wma_200","position","previous_position"]]

# -------------------- Run --------------------
def run():
    pairs = select_pairs(MUST_HAVE_BASES, take_total=30)
    bases = [p.replace("USDT", "") for p in pairs]

    all_rows: List[Dict] = []
    alerts = 0
    latest_dates: Dict[str, str] = {}

    for base in bases:
        log(f"• Processing {base}")
        try:
            daily = build_daily_for_symbol(base, DAYS_BACK)
            if daily is None or daily.empty:
                log(f"  └─ no data for {base}")
                continue

            # Telegram: if latest position changed vs previous
            if len(daily) >= 2:
                latest = daily.iloc[-1]
                prev   = daily.iloc[-2]
                if latest["position"] != prev["position"]:
                    msg = (
                        f"{base} changed position: {prev['position']} → {latest['position']}\n"
                        f"Close: {float(latest['close']):.6g}"
                    )
                    if send_telegram(msg):
                        alerts += 1
                        log(f"  └─ ✅ Telegram alert sent for {base}")

            latest_dates[base] = daily.iloc[-1]["date"]

            # JSON-safe rows (NaN → None)
            clean = daily.where(pd.notna(daily), None)
            all_rows.extend(clean.to_dict(orient="records"))

        except Exception as e:
            log(f"  └─ ❌ {base}: {e}")

    if not all_rows:
        log("No rows to upsert.")
        return

    # Dedup (coin,date)
    ded = {}
    for r in all_rows:
        ded[(r["coin"], r["date"])] = r
    payload = list(ded.values())

    log(f"⬆️ Upserting {len(payload)} rows to Supabase…")
    supabase.table("coin_wma").upsert(payload, on_conflict="coin,date", returning="minimal").execute()
    log("✅ Supabase upsert complete")

    # small summary
    max_dt = max(latest_dates.values()) if latest_dates else "—"
    log(f"Coins updated: {len(latest_dates)} | Most recent date: {max_dt} | Telegram alerts: {alerts}")

if __name__ == "__main__":
    # make logs unbuffered in GH Actions too
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
