#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py  (Hourly, incremental + auto-backfill + Telegram)

- Guarantees your must-have list (normalizes 1000 SHIB->SHIB, UNIS->UNI; HYPE is skipped if not on Binance)
- Adds extra coins from CoinGecko to fill to 30 Binance-tradable USDT pairs
- Uses Binance HOURLY klines, then picks the candle closest to 12:00 Europe/Istanbul per day
- INCREMENTAL: fetches only missing days; if a coin has < 220 rows in DB, it auto-backfills
- Computes WMA(50) and WMA(200) with strict windows; classifies position / previous_position
- Upserts to Supabase with conflict target (coin,date)
- Telegram alerts when a coin’s Position changes vs the prior day

Env (required):  SUPABASE_URL, SUPABASE_KEY
Env (optional):  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import time
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from supabase import create_client, Client

# ---------- Config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SESSION = requests.Session()
IST = ZoneInfo("Europe/Istanbul")

MIN_ROWS_FOR_WMA200 = 220   # rows per coin we target to ensure WMA200 is computable

# ---------- Telegram ----------
def send_telegram_alert(message: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("ℹ️ Telegram not configured; skipping alert.")
        return
    try:
        resp = SESSION.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=15,
        )
        if resp.status_code != 200:
            print("❌ Telegram API error:", resp.status_code, resp.text)
        else:
            print("✅ Telegram alert sent")
    except Exception as e:
        print("❌ Telegram send failed:", e)

# ---------- CoinGecko ----------
def get_top_symbols_for_headroom(limit: int = 200) -> list[str]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return [item["symbol"].upper() for item in resp.json()]

# ---------- Binance ----------
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

def fetch_hourly_klines(pair: str, start_ts_ms: int, end_ts_ms: int):
    """Fetch hourly klines in [start,end)."""
    out, curr, it, backoff = [], start_ts_ms, 0, 0.2
    while curr < end_ts_ms and it < 200:
        hours_needed = int((end_ts_ms - curr) / (3600 * 1000)) + 1
        limit = min(1000, max(1, hours_needed))
        params = {"symbol": pair, "interval": "1h", "startTime": curr, "limit": limit}
        try:
            r = SESSION.get(BINANCE_KLINES_URL, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff); backoff = min(backoff * 2, 4.0); continue
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            out.extend(data)
            last_open = data[-1][0]
            if last_open <= curr:
                break
            curr = last_open + 3600 * 1000
            time.sleep(0.08)
        except Exception as e:
            print(f"❌ Error fetching klines for {pair}: {e}")
            time.sleep(0.5)
            break
        it += 1
    return out

# ---------- Resample: hourly → “noon” ----------
def hourly_to_noon_daily_close(hourly_df: pd.DataFrame):
    """Pick candle closest to 12:00 Europe/Istanbul per local day."""
    df = hourly_df.copy()
    df["local_time"] = df["open_time"].dt.tz_convert("Europe/Istanbul")
    df["local_date"] = df["local_time"].dt.date
    df["hour_diff"] = (df["local_time"].dt.hour - 12).abs()
    idx = df.groupby("local_date")["hour_diff"].idxmin()
    daily = df.loc[idx].copy()
    daily["date"] = pd.to_datetime(daily["local_date"])
    return daily[["date", "close"]].sort_values("date").reset_index(drop=True)

# ---------- Indicators ----------
def wma(series: pd.Series, window: int):
    s = series.dropna()
    if len(s) < window: return np.nan
    weights = np.arange(1, window + 1, dtype=float)
    return float((s.tail(window).to_numpy() * weights).sum() / weights.sum())

def classify_position(close, wma50, wma200):
    if close is None or pd.isna(close) or pd.isna(wma50) or pd.isna(wma200):
        return "Not enough data"
    if close > max(wma50, wma200): return "Above both"
    if close < min(wma50, wma200): return "Below both"
    return "Between"

# ---------- Supabase helpers ----------
def get_latest_dates_from_supabase() -> dict[str, str]:
    """{coin: 'YYYY-MM-DD'} for last date per coin."""
    try:
        resp = supabase.table("coin_wma").select("coin,date").execute()
        rows = resp.data or []
        df = pd.DataFrame(rows)
        if df.empty: return {}
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest = df.sort_values(["coin","date"]).groupby("coin").tail(1)
        return {row["coin"]: row["date"].date().isoformat() for _, row in latest.iterrows()}
    except Exception as e:
        print("Could not load latest dates from Supabase:", e)
        return {}

def get_row_count_from_supabase(coin: str) -> int:
    """How many rows we already have for this coin? (used to auto-backfill if < 220)"""
    try:
        resp = supabase.table("coin_wma").select("date", count="exact").eq("coin", coin).execute()
        return int(getattr(resp, "count", 0) or 0)
    except Exception:
        return 0

def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    return df.replace({np.nan: None})

def to_native_types(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({k: (float(v) if isinstance(v, (np.floating,))
                        else int(v) if isinstance(v, (np.integer,))
                        else v) for k, v in r.items()})
    return out

# ---------- Incremental date planning ----------
def dates_to_fetch_for_coin(coin: str, latest_map: dict[str, str], backfill_days: int = MIN_ROWS_FOR_WMA200) -> list[date]:
    """
    If the coin has < backfill_days rows total, fetch the last `backfill_days` local days.
    Else, fetch from (last_date+1) to today.
    """
    today_local = datetime.now(tz=IST).date()
    current_rows = get_row_count_from_supabase(coin)

    if current_rows < backfill_days:
        start = today_local - pd.Timedelta(days=backfill_days - 1)
        return pd.date_range(start, today_local, freq="D").date.tolist()

    if coin in latest_map:
        last = pd.to_datetime(latest_map[coin]).date()
        start = last + pd.Timedelta(days=1)
        if start > today_local:
            return []
        return pd.date_range(start, today_local, freq="D").date.tolist()

    start = today_local - pd.Timedelta(days=backfill_days - 1)
    return pd.date_range(start, today_local, freq="D").date.tolist()

# ---------- Hourly → noon for a set of dates ----------
def get_noon_series_for_dates_hourly(symbol: str, dates_local: list[date]) -> pd.DataFrame | None:
    if not dates_local: return None
    pair = f"{symbol}USDT"

    d_min, d_max = min(dates_local), max(dates_local)
    start_local = datetime(d_min.year, d_min.month, d_min.day, 0, 0, tzinfo=IST) - pd.Timedelta(days=2)
    end_local   = datetime(d_max.year, d_max.month, d_max.day, 23, 59, tzinfo=IST) + pd.Timedelta(days=1)
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    end_ms   = int(end_local.astimezone(timezone.utc).timestamp() * 1000)

    raw = fetch_hourly_klines(pair, start_ms, end_ms)
    if not raw: return None

    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open_time","close"])

    daily = hourly_to_noon_daily_close(df)
    if daily is None or daily.empty: return None

    mask = daily["date"].dt.date.isin(dates_local)
    out = daily.loc[mask].copy().sort_values("date").reset_index(drop=True)
    return out if not out.empty else None

# ---------- Main ----------
def run_pipeline():
    # Must-have list
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

    # Fill to 30
    bases_extra = get_top_symbols_for_headroom(limit=200)
    all_bases = must_have + [b for b in bases_extra if b not in must_have]
    pairs_all = map_to_valid_pair(all_bases, "USDT")
    missing_must = [b for b in must_have if f"{b}USDT" not in set(pairs_all)]
    if missing_must:
        print("Must-have symbols NOT available on Binance (skipped):", missing_must)
    pairs = pairs_all[:30]
    print(f"✅ Tradable Binance pairs (up to 30, must-haves first): {pairs}")

    latest_map = get_latest_dates_from_supabase()
    all_rows: list[dict] = []

    for pair in pairs:
        sym = pair.replace("USDT", "")
        try:
            need_dates = dates_to_fetch_for_coin(sym, latest_map, backfill_days=MIN_ROWS_FOR_WMA200)
            if not need_dates:
                print(f"{sym}: up to date; skipping.")
                continue

            daily_new = get_noon_series_for_dates_hourly(sym, need_dates)
            if daily_new is None or daily_new.empty:
                print(f"{sym}: no data for needed dates; skipping.")
                continue

            # Pull prior window to make WMAs correct around the boundary
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

            # Indicators
            daily["WMA_50"]  = daily["close"].rolling(50,  min_periods=50).apply(lambda s: wma(s, 50),  raw=False)
            daily["WMA_200"] = daily["close"].rolling(200, min_periods=200).apply(lambda s: wma(s, 200), raw=False)
            daily["Position"] = daily.apply(lambda r: classify_position(r["close"], r["WMA_50"], r["WMA_200"]), axis=1)
            daily["Previous_Position"] = daily["Position"].shift(1)

            # Only the new dates get upserted
            mask_new = daily["date"].dt.date.isin(need_dates)
            out = daily.loc[mask_new, ["date","close","WMA_50","WMA_200","Position","Previous_Position"]].copy()
            out["coin"] = sym
            out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
            out = out.rename(columns={"WMA_50": "wma_50", "WMA_200": "wma_200",
                                      "Position": "position", "Previous_Position": "previous_position"})
            out = clean_for_json(out)
            all_rows.extend(to_native_types(out.to_dict(orient="records")))

            # Alert only if the last row we inserted is among need_dates and changed vs previous
            if len(daily) >= 2:
                last_day = pd.to_datetime(daily.iloc[-1]["date"]).date()
                if last_day in need_dates:
                    prev, latest = daily.iloc[-2], daily.iloc[-1]
                    if latest["Position"] != prev["Position"]:
                        msg = f"{sym} changed position: {prev['Position']} → {latest['Position']}\nClose: {latest['close']}"
                        send_telegram_alert(msg)

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")

    if not all_rows:
        print("Nothing new to upload — all up to date.")
        return

    # Upsert
    try:
        dedup = {}
        for r in all_rows:
            dedup[(r["coin"], r["date"])] = r
        all_rows = list(dedup.values())
        print(f"⬆️ Upserting {len(all_rows)} rows to Supabase…")
        supabase.table("coin_wma").upsert(all_rows, on_conflict="coin,date", returning="minimal").execute()
        print("✅ Supabase upsert complete")
    except Exception as e:
        print("❌ Supabase upload failed:", e)

if __name__ == "__main__":
    run_pipeline()
