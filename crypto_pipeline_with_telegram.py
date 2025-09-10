#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py

- Must-have coin list is attempted first (with symbol normalization)
- Add extra coins from CoinGecko (200) and keep first 30 Binance-tradable USDT pairs
- Use Binance 1-minute klines to capture the EXACT 12:00 Europe/Istanbul close per day
- Incremental updates: fetch only missing days per coin (fast after first backfill)
- Compute WMA(50) and WMA(200) with strict windows
- Classify position and previous_position
- Upsert into Supabase (coin_wma) with conflict target (coin,date)
- Send Telegram alerts on position change (latest vs previous)

Env:
  SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN (optional), TELEGRAM_CHAT_ID (optional)
"""

import os
import time
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from supabase import create_client, Client

# -------------------------- Config --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SESSION = requests.Session()
IST = ZoneInfo("Europe/Istanbul")

# -------------------------- Telegram ------------------------
def send_telegram_alert(message: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram not configured; skipping alert.")
        return
    try:
        resp = SESSION.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10,
        )
        if resp.status_code != 200:
            print("Telegram API returned", resp.status_code, resp.text)
    except Exception as e:
        print("Failed to send Telegram message:", e)

# -------------------------- CoinGecko -----------------------
def get_top_symbols_for_headroom(limit: int = 200) -> list[str]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return [item["symbol"].upper() for item in resp.json()]

# -------------------------- Binance -------------------------
BINANCE_API = "https://api.binance.com"
BINANCE_KLINES_URL = f"{BINANCE_API}/api/v3/klines"

_BINANCE_SYMBOLS_CACHE: set[str] | None = None

def get_binance_spot_symbols() -> set[str]:
    global _BINANCE_SYMBOLS_CACHE
    if _BINANCE_SYMBOLS_CACHE is not None:
        return _BINANCE_SYMBOLS_CACHE
    r = SESSION.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    info = r.json()
    _BINANCE_SYMBOLS_CACHE = {
        s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"
    }
    return _BINANCE_SYMBOLS_CACHE

def map_to_valid_pair(bases: list[str], quote: str = "USDT") -> list[str]:
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

def fetch_minute_klines_exact(pair: str, start_ms: int, limit: int = 1):
    """
    Fetch 1-minute klines starting at start_ms (UTC ms).
    With limit=1, Binance returns the candle whose open >= start_ms.
    We validate exact open_time (== start_ms) afterward.
    """
    params = {"symbol": pair, "interval": "1m", "startTime": start_ms, "limit": limit}
    r = SESSION.get(BINANCE_KLINES_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# -------------------------- Supabase helpers ---------------
def get_latest_dates_from_supabase() -> dict[str, str]:
    """
    Returns { 'BTC': 'YYYY-MM-DD', ... } for coins present in coin_wma.
    """
    try:
        resp = supabase.table("coin_wma").select("coin,date").execute()
        rows = resp.data or []
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest = df.sort_values(["coin", "date"]).groupby("coin").tail(1)
        return {row["coin"]: row["date"].date().isoformat() for _, row in latest.iterrows()}
    except Exception as e:
        print("Could not load latest dates from Supabase:", e)
        return {}

# -------------------------- Date planning -------------------
def dates_to_fetch_for_coin(coin: str, days_backfill: int = 220, latest_map: dict | None = None) -> list[date]:
    """
    If coin has data: fetch from (last_date+1) .. today (Europe/Istanbul).
    Else: backfill last `days_backfill` days (enough for WMA200).
    """
    today_local = datetime.now(tz=IST).date()
    if latest_map and coin in latest_map:
        last = pd.to_datetime(latest_map[coin]).date()
        start = last + pd.Timedelta(days=1)
        if start > today_local:
            return []
        rng = pd.date_range(start, today_local, freq="D").date.tolist()
        return rng
    # no history -> initial backfill
    start = today_local - pd.Timedelta(days=days_backfill - 1)
    return pd.date_range(start, today_local, freq="D").date.tolist()

# -------------------------- Noon close fetch ---------------
def get_noon_series_for_dates(symbol: str, dates_local: list[date]) -> pd.DataFrame | None:
    """
    For a given symbol and list of *local Istanbul* dates, fetch the exact 12:00 close (1m kline).
    Returns DataFrame ['date','close'] (date as pandas.Timestamp normalized to date).
    """
    pair = f"{symbol}USDT"
    rows = []
    for d in sorted(set(dates_local)):
        ts_local = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=IST)
        ts_utc = ts_local.astimezone(timezone.utc)
        start_ms = int(ts_utc.timestamp() * 1000)
        try:
            data = fetch_minute_klines_exact(pair, start_ms=start_ms, limit=1)
            got_exact = False
            if data:
                if data[0][0] == start_ms:
                    rows.append({"date": pd.Timestamp(d), "close": float(data[0][4])})
                    got_exact = True
            if not got_exact:
                # Fallback: small window around 12:00 (11:59–12:01) to find the exact minute
                window_start = start_ms - 60_000
                data2 = fetch_minute_klines_exact(pair, start_ms=window_start, limit=3)
                found = False
                for k in data2 or []:
                    if k[0] == start_ms:
                        rows.append({"date": pd.Timestamp(d), "close": float(k[4])})
                        found = True
                        break
                if not found:
                    print(f"[{pair}] Missing exact 12:00 local minute for {d}")
            time.sleep(0.03)  # small pause
        except Exception as e:
            print(f"[{pair}] error on {d}: {e}")
            time.sleep(0.2)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# -------------------------- Indicators ---------------------
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

# -------------------------- JSON cleaning ------------------
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

# -------------------------- Main pipeline ------------------
def run_pipeline(days=250):
    # 1) Must-have bases (normalized)
    must_have_raw = [
        "BTC","ETH","XRP","BNB","SOL","DOGE","TRX","ADA",
        "HYPE","XLM","SUI","LINK","BCH","HBAR","AVAX","TON",
        "LTC","1000 SHIB","DOT","UNIS"
    ]
    normalize = {"1000SHIB": "SHIB", "UNIS": "UNI"}
    def norm(sym: str) -> str:
        s = sym.upper().replace(" ", "")
        return normalize.get(s, s)

    must_have = [norm(s) for s in must_have_raw]

    # 2) Add extra bases from CoinGecko to fill to 30
    bases_extra = get_top_symbols_for_headroom(limit=200)
    all_bases = must_have + [b for b in bases_extra if b not in must_have]

    # 3) Map to Binance tradable USDT pairs
    pairs_all = map_to_valid_pair(all_bases, "USDT")
    listed_pairs_set = set(pairs_all)
    missing_must = [b for b in must_have if f"{b}USDT" not in listed_pairs_set]
    if missing_must:
        print("Must-have symbols NOT available on Binance (skipped):", missing_must)

    # Keep first 30 tradable pairs (must-haves appear first)
    pairs = pairs_all[:30]
    print(f"Tradable Binance pairs (up to 30, must-haves first): {pairs}")

    # 4) Incremental plan: figure out which dates each coin needs
    latest_map = get_latest_dates_from_supabase()
    all_rows: list[dict] = []

    for pair in pairs:
        sym = pair.replace("USDT", "")
        try:
            need_dates = dates_to_fetch_for_coin(sym, days_backfill=220, latest_map=latest_map)
            if not need_dates:
                print(f"{sym} is up to date; skipping.")
                continue

            daily_new = get_noon_series_for_dates(sym, need_dates)
            if daily_new is None or daily_new.empty:
                print("No data for", sym)
                continue

            # For correct rolling WMAs, pull up to 199 prior days from Supabase (light query)
            try:
                prev_start = (min(need_dates) - pd.Timedelta(days=199)).isoformat()
                resp_prev = supabase.table("coin_wma")\
                    .select("date,close")\
                    .eq("coin", sym)\
                    .gte("date", prev_start)\
                    .order("date")\
                    .execute()
                prev_df = pd.DataFrame(resp_prev.data or [])
                if not prev_df.empty:
                    prev_df["date"] = pd.to_datetime(prev_df["date"], errors="coerce")
                    prev_df["close"] = pd.to_numeric(prev_df["close"], errors="coerce")
                    daily = pd.concat([prev_df[["date","close"]], daily_new], ignore_index=True)\
                               .drop_duplicates(subset=["date"]).sort_values("date")
                else:
                    daily = daily_new
            except Exception as e:
                print(f"{sym}: could not load prior window from Supabase: {e}")
                daily = daily_new

            # Indicators (strict windows)
            daily["WMA_50"]  = daily["close"].rolling(window=50,  min_periods=50).apply(lambda s: wma(s, 50),  raw=False)
            daily["WMA_200"] = daily["close"].rolling(window=200, min_periods=200).apply(lambda s: wma(s, 200), raw=False)
            daily["Position"] = daily.apply(lambda r: classify_position(r["close"], r["WMA_50"], r["WMA_200"]), axis=1)
            daily["Previous_Position"] = daily["Position"].shift(1)

            # Only upsert the *new* dates we just fetched
            mask_new = daily["date"].dt.date.isin(need_dates)
            out = daily.loc[mask_new, ["date","close","WMA_50","WMA_200","Position","Previous_Position"]].copy()
            out["coin"] = sym
            out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
            out = out.rename(columns={
                "WMA_50": "wma_50",
                "WMA_200": "wma_200",
                "Position": "position",
                "Previous_Position": "previous_position",
            })
            out = clean_for_json(out)
            all_rows.extend(to_native_types(out.to_dict(orient="records")))

            # Alert only if latest day was part of need_dates and we have 2 most recent rows
            dtail = daily.tail(2)
            if len(dtail) == 2 and dtail.iloc[-1]["date"].date() in need_dates:
                if dtail.iloc[-1]["Position"] != dtail.iloc[-2]["Position"]:
                    msg = f"{sym} changed position: {dtail.iloc[-2]['Position']} → {dtail.iloc[-1]['Position']}\nClose: {dtail.iloc[-1]['close']}"
                    send_telegram_alert(msg)

        except Exception as e:
            print(f"Error processing {sym}: {e}")

    if not all_rows:
        print("Nothing new to upload — all up to date.")
        return

    # 5) Upsert
    try:
        # Deduplicate batch on (coin,date)
        dedup = {}
        for r in all_rows:
            dedup[(r["coin"], r["date"])] = r
        all_rows = list(dedup.values())

        print(f"Upserting {len(all_rows)} rows to Supabase...")
        supabase.table("coin_wma").upsert(
            all_rows, on_conflict="coin,date", returning="minimal"
        ).execute()
        print("Upsert complete.")
    except Exception as e:
        print("Supabase upload failed:", e)

# -------------------------- Entrypoint ---------------------
if __name__ == "__main__":
    run_pipeline()
