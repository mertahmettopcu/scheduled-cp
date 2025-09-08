 #!/usr/bin/env python3
"""
 crypto_pipeline_with_telegram.py- Fetch top 30 coins from CoinGecko- Download hourly klines from Binance, pick candle closest to 12:00 Europe/Istanbul each day- Compute WMA(50) and WMA(200)- Classify position and previous_position- Upsert into Supabase (coin_wma table)- Send Telegram alerts on position change (latest vs previous)
 Requirements: set environment variables SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
 Run: python crypto_pipeline_with_telegram.py
 """
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client, Client
# -------------------------- Config (via environment variables) --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Service role key recommended for insert/upsert
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not (SUPABASE_URL and SUPABASE_KEY):
   raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY env vars. See guide.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# -------------------------- Helpers -----------------------------------------------------
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
 # -------------------------- CoinGecko top 30 -------------------------------------------
def get_top_30():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10, "page": 1}  ## cahnge per_page back to 30 later on
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Return symbols in uppercase (we will try SYMBOL+USDT on Binance)
    return [item["symbol"].upper() for item in data]
 # -------------------------- Binance hourly kline fetch with pagination -----------------
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
def fetch_hourly_klines(pair: str, start_ts_ms: int, end_ts_ms: int):
    """Fetch hourly klines for pair between start_ts_ms and end_ts_ms (ms epoch).
    Binance allows max 1000 candles per request, so we paginate by moving startTime forward.
    Returns list of kline records (raw arrays).
    """
    out = []
    curr = start_ts_ms
    max_iters = 200  # safety
    it = 0
    while curr < end_ts_ms and it < max_iters:
        # compute max number of candles we need from curr to end, capped at 1000
        ms_range = end_ts_ms - curr
        hours_needed = int(ms_range / (3600 * 1000)) + 1
        limit = min(1000, hours_needed)
        #params = {"symbol": pair, "interval": "1h", "startTime": curr, "limit": limit}
        params = {"symbol": pair, "interval": "1h", "limit": limit}

        try:
            r = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
            print(f"Status Code: {r.status_code}") ##### you may remove this line ##########################################################
            print(f"Raw response for {pair}: {r.text[:500]}") ##### you may remove this line ##########################################################
            r.raise_for_status()
            data = r.json()
            if not data:
                print(f"No data for {pair}") ##### you may remove this line ##########################################################
                break
            out.extend(data)
            print(f"Fetched {len(data)} candels for {pair}") ##### you may remove this line ##########################################
            last_open = data[-1][0]  # open time ms of last candle returned
            curr = last_open + 3600 * 1000  # move to next hour after last_open
            time.sleep(0.12)  # be polite to Binance
        except Exception as e:
            print("Error fetching klines for", pair, e)
            time.sleep(1)
            break
        it += 1
    return out
 # -------------------------- Select candle closest to 12:00 Europe/Istanbul each day ----
def hourly_to_noon_daily_close(hourly_df: pd.DataFrame):
    """Given hourly dataframe with 'open_time' (tz-aware UTC) and 'close' as float,
    convert to Europe/Istanbul timezone and pick the hourly candle closest to 12:00 local time
    for each local calendar day."""
    df = hourly_df.copy()
    # convert open_time (UTC) to Europe/Istanbul tz
    df["local_time"] = df["open_time"].dt.tz_convert("Europe/Istanbul")
    df["local_date"] = df["local_time"].dt.date
    df["hour_diff"] = (df["local_time"].dt.hour - 12).abs()
    idx = df.groupby("local_date")["hour_diff"].idxmin()
    daily = df.loc[idx].copy()
    daily["date"] = pd.to_datetime(daily["local_date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily[["date", "close"]]
 # -------------------------- WMA calculation and classification ------------------------
def wma(series: pd.Series, window: int):
    weights = range(1, window + 1)
    return (series * weights).sum() / sum(weights)
def classify_position(close, wma50, wma200):
    import math
    if wma50 is None or wma200 is None or (pd.isna(wma50) or pd.isna(wma200)):
        return "Not enough data"
    lower = min(wma50, wma200)
    upper = max(wma50, wma200)
    if close > upper:
        return "Above both"
    elif close < lower:
        return "Below both"
    else:
        return "Between"
 # -------------------------- Main pipeline ---------------------------------------------
def get_binance_noon_series(symbol: str, days: int = 250):
    """Return a DataFrame with hourly->noon daily series for the last `days` days
    for symbol (example: BTC -> BTCUSDT)."""
    pair = symbol + "USDT"
    end_ts = int(datetime.utcnow().timestamp() * 1000)
    start_ts = end_ts - int((days + 10) * 24 * 3600 * 1000)  # a little extra margin
    raw = fetch_hourly_klines(pair, start_ts, end_ts)
    if not raw:
        return None
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    daily = hourly_to_noon_daily_close(df)
    return daily
def run_pipeline(days=250):
    symbols = get_top_30()
    print("Top symbols:", symbols)
    all_rows = []
    for sym in symbols:
        try:
            print("Processing", sym)
            daily = get_binance_noon_series(sym, days=days)
            if daily is None or daily.empty:
                print("No data for", sym)
                continue
            daily = daily.sort_values("date").reset_index(drop=True)
            # compute WMA
            daily["WMA_50"] = daily["close"].rolling(50).apply(lambda s: wma(s, 50), raw=True)
            daily["WMA_200"] = daily["close"].rolling(200).apply(lambda s: wma(s, 200), raw=True)
            daily["Position"] = daily.apply(lambda r: classify_position(r["close"], r["WMA_50"], r["WMA_200"]), axis=1)
            daily["Previous_Position"] = daily["Position"].shift(1)
            daily["coin"] = sym
            # Prepare rows for upsert: keep date as ISO date string
            rows = daily[["coin","date","close","WMA_50","WMA_200","Position","Previous_Position"]].copy()
            rows["date"] = rows["date"].dt.date.astype(str)
            all_rows.extend(rows.to_dict(orient="records"))
            # Telegram alert for latest change
            if len(daily) >= 2:
                latest = daily.iloc[-1]
                previous = daily.iloc[-2]
                if latest["Position"] != previous["Position"]:
                    msg = f"{sym} changed position: {previous['Position']} → {latest['Position']}\\nClose: {latest['close']}"
                    send_telegram_alert(msg)
        except Exception as e:
            print("Error processing", sym, e)
            continue
    if not all_rows:
        print("No rows to upload")
        return
    # Upsert into Supabase (note: ensure unique index on (coin, date) in Supabase)
    try:
        print("Upserting", len(all_rows), "rows to Supabase...")
        resp = supabase.table("coin_wma").upsert(all_rows).execute()
        print("Supabase response:", getattr(resp, "status_code", None), resp)
    except Exception as e:
        print("Supabase upload failed:", e)
if __name__ == "__main__":
    run_pipeline()
