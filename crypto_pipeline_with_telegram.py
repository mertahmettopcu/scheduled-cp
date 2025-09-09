#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py
- Ensure a must-have coin list is attempted first
- Fetch extra coins from CoinGecko to fill to 30 total
- Download hourly klines from Binance, pick candle closest to 12:00 Europe/Istanbul each day
- Compute WMA(50) and WMA(200)
- Classify position and previous_position
- Upsert into Supabase (coin_wma table)
- Send Telegram alerts on position change (latest vs previous)

Env:
  SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client, Client

# -------------------------- Config (via environment variables) --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Prefer anon for read-only; service role only on trusted servers
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY env vars. See guide.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------- Telegram helper --------------------------------------------
def send_telegram_alert(message: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram not configured; skipping alert.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            print("Telegram API returned", resp.status_code, resp.text)
    except Exception as e:
        print("Failed to send Telegram message:", e)

# -------------------------- CoinGecko --------------------------------------------
def get_top_symbols_for_headroom(limit: int = 200) -> list[str]:
    """
    Ask CoinGecko for many coins (default 200) to have slack,
    then we keep the first 30 Binance-valid after must-haves.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return [item["symbol"].upper() for item in data]

# -------------------------- Binance helpers --------------------------------------------
BINANCE_API = "https://api.binance.com"
BINANCE_KLINES_URL = f"{BINANCE_API}/api/v3/klines"

def get_binance_spot_symbols() -> set[str]:
    r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    info = r.json()
    return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

def map_to_valid_pair(bases: list[str], quote: str = "USDT") -> list[str]:
    """
    Return only valid spot pairs on Binance. Skips self-quoted (e.g., USDT/USDT).
    """
    listed = get_binance_spot_symbols()
    out = []
    for b in bases:
        if b.upper() == quote.upper():
            print(f"Skipping self-quoted pair: {b}{quote}")
            continue
        sym = f"{b.upper()}{quote.upper()}"
        if sym in listed:
            out.append(sym)
        else:
            print(f"Skipping unsupported pair: {sym}")
    return out

def fetch_hourly_klines(pair: str, start_ts_ms: int, end_ts_ms: int):
    """
    Fetch hourly klines for pair between start_ts_ms and end_ts_ms (ms epoch).
    Binance allows max 1000 candles per request; we paginate using startTime.
    """
    out = []
    curr = start_ts_ms
    max_iters = 200
    it = 0
    backoff = 0.25

    while curr < end_ts_ms and it < max_iters:
        hours_needed = int((end_ts_ms - curr) / (3600 * 1000)) + 1
        limit = min(1000, max(hours_needed, 1))
        params = {
            "symbol": pair,
            "interval": "1h",
            "startTime": curr,
            "endTime": end_ts_ms,
            "limit": limit,
        }
        try:
            r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
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
            print(f"Error fetching klines for {pair}: {e}")
            time.sleep(1)
            break
        it += 1
    return out

# -------------------------- Timezone & resampling ---------------------------------------
def hourly_to_noon_daily_close(hourly_df: pd.DataFrame):
    df = hourly_df.copy()
    df["local_time"] = df["open_time"].dt.tz_convert("Europe/Istanbul")
    df["local_date"] = df["local_time"].dt.date
    df["hour_diff"] = (df["local_time"].dt.hour - 12).abs()
    idx = df.groupby("local_date")["hour_diff"].idxmin()
    daily = df.loc[idx].copy()
    daily["date"] = pd.to_datetime(daily["local_date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily[["date", "close"]]

# -------------------------- Indicators & classification ---------------------------------
def wma(series: pd.Series, window: int):
    s = series.dropna()
    if len(s) < window:
        return np.nan
    weights = np.arange(1, window + 1, dtype=float)
    return float((s.tail(window).to_numpy() * weights).sum() / weights.sum())

def classify_position(close, wma50, wma200):
    if close is None or pd.isna(close) or pd.isna(wma50) or pd.isna(wma200):
        return "Not enough data"
    lower = min(wma50, wma200)
    upper = max(wma50, wma200)
    if close > upper:
        return "Above both"
    elif close < lower:
        return "Below both"
    else:
        return "Between"

# -------------------------- Cleaning for JSON / Supabase --------------------------------
def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number, "number", "float64", "int64"]).columns.tolist()
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    return df.replace({np.nan: None})

def to_native_types(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            rr[k] = v
        out.append(rr)
    return out

# -------------------------- Main pipeline -----------------------------------------------
def get_binance_noon_series(symbol: str, days: int = 250):
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = end_ts - int((days + 10) * 24 * 3600 * 1000)
    pair = f"{symbol}USDT"

    raw = fetch_hourly_klines(pair, start_ts, end_ts)
    if not raw:
        return None

    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","taker_base","taker_quote","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open_time", "close"])
    return hourly_to_noon_daily_close(df)

def run_pipeline(days=250):
    # --------- 1) Must-have bases (normalized) ----------
    must_have_raw = [
        "BTC","ETH","XRP","BNB","SOL","DOGE","TRX","ADA",
        "HYPE","XLM","SUI","LINK","BCH","HBAR","AVAX","TON",
        "LTC","1000 SHIB","DOT","UNIS"
    ]

    # Normalize known variants to Binance-style bases
    normalize = {
        "1000 SHIB": "SHIB",
        "1000SHIB": "SHIB",
        "UNIS": "UNI",
    }
    must_have = [normalize.get(sym.upper().replace(" ", ""), sym.upper().replace(" ", "")) for sym in must_have_raw]
    # Note: "HYPE" may not exist on Binance; we will log it if missing.

    # --------- 2) Add extra bases from CoinGecko to fill to 30 ----------
    bases_extra = get_top_symbols_for_headroom(limit=200)
    # Keep order: must-haves first, then extras not already in must-have
    all_bases = must_have + [b for b in bases_extra if b not in must_have]

    # --------- 3) Map to Binance tradable USDT pairs ----------
    pairs_all = map_to_valid_pair(all_bases, "USDT")

    # Log any must-haves that did not make it to Binance pairs (unsupported or self-quote)
    listed_pairs_set = set(pairs_all)
    missing_must = [b for b in must_have if f"{b}USDT" not in listed_pairs_set]
    if missing_must:
        print("Must-have symbols NOT available on Binance (skipped):", missing_must)

    # Keep exactly the first 30 tradable pairs
    pairs = pairs_all[:30]
    print(f"Tradable Binance pairs (up to 30, must-haves first): {pairs}")

    # --------- 4) Process & upsert ----------
    all_rows = []

    for pair in pairs:
        sym = pair.replace("USDT", "")
        try:
            print("Processing", sym)
            daily = get_binance_noon_series(sym, days=days)
            if daily is None or daily.empty:
                print("No data for", sym)
                continue

            daily = daily.sort_values("date").reset_index(drop=True)

            daily["WMA_50"] = daily["close"].rolling(window=50, min_periods=50)\
                                            .apply(lambda s: wma(s, 50), raw=False)
            daily["WMA_200"] = daily["close"].rolling(window=200, min_periods=200)\
                                             .apply(lambda s: wma(s, 200), raw=False)

            daily["Position"] = daily.apply(
                lambda r: classify_position(r["close"], r["WMA_50"], r["WMA_200"]),
                axis=1
            )
            daily["Previous_Position"] = daily["Position"].shift(1)

            daily = daily.dropna(subset=["close", "WMA_50", "WMA_200"]).reset_index(drop=True)

            out = daily[["date", "close", "WMA_50", "WMA_200", "Position", "Previous_Position"]].copy()
            out["coin"] = sym
            out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)

            out = out.rename(columns={
                "WMA_50": "wma_50",
                "WMA_200": "wma_200",
                "Position": "position",
                "Previous_Position": "previous_position",
            })

            out = clean_for_json(out)
            rows = to_native_types(out.to_dict(orient="records"))
            all_rows.extend(rows)

            # Alerts
            if len(daily) >= 2:
                latest = daily.iloc[-1]
                previous = daily.iloc[-2]
                if latest["Position"] != previous["Position"]:
                    msg = f"{sym} changed position: {previous['Position']} → {latest['Position']}\nClose: {latest['close']}"
                    send_telegram_alert(msg)

        except requests.HTTPError as e:
            print(f"HTTP error on {sym}: {e}")
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    if not all_rows:
        print("No rows to upload")
        return

    try:
        # Deduplicate within the batch on (coin, date)
        dedup = {}
        for r in all_rows:
            dedup[(r["coin"], r["date"])] = r
        all_rows = list(dedup.values())

        print(f"Upserting {len(all_rows)} rows to Supabase...")
        resp = supabase.table("coin_wma").upsert(
            all_rows,
            on_conflict="coin,date",
            returning="minimal"
        ).execute()
        data_len = len(resp.data) if getattr(resp, "data", None) else None
        print(f"Supabase upsert OK. Rows echoed: {data_len}")
    except Exception as e:
        print("Supabase upload failed:", e)

if __name__ == "__main__":
    run_pipeline()
