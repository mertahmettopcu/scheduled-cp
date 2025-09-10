#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py  (HOURLY • INCREMENTAL • DATE-SPAN BACKFILL • TELEGRAM)

- Guarantees your must-have list (normalizes 1000 SHIB->SHIB, UNIS->UNI; HYPE skipped if not listed)
- Adds extra CoinGecko symbols to fill to 30 Binance USDT pairs
- Fetches HOURLY klines; picks candle closest to 12:00 Europe/Istanbul per local day
- INCREMENTAL by **date span**:
    • If no data: fetch last 220 days (enough for WMA200)
    • If max(date) < today: fetch missing forward days
    • If coverage < 220 days ending today: fetch older days to complete the span
- Computes WMA(50), WMA(200), Position/Previous_Position
- Upserts to Supabase on (coin,date)
- Sends Telegram alerts when Position changes (logs which coin)
"""

import os
import time
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from supabase import create_client, Client

# ---------------- Config ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SESSION = requests.Session()
IST = ZoneInfo("Europe/Istanbul")

MIN_ROWS_FOR_WMA200 = 220  # target rolling coverage window (days)

# ------------- Telegram -------------
def send_telegram_alert(message: str, coin: str | None = None):
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
            print(f"❌ Telegram API error ({coin or 'unknown'}):", resp.status_code, resp.text)
        else:
            print(f"✅ Telegram alert sent for {coin or 'unknown'}")
    except Exception as e:
        print(f"❌ Telegram send failed ({coin or 'unknown'}):", e)

# ------------- CoinGecko -------------
def get_top_symbols_for_headroom(limit: int = 200) -> list[str]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return [item["symbol"].upper() for item in r.json()]

# ------------- Binance -------------
BINANCE_API = "https://api.binance.com"
BINANCE_KLINES_URL = f"{BINANCE_API}/api/v3/klines"
_SYMBOLS_CACHE: set[str] | None = None

def get_binance_spot_symbols() -> set[str]:
    global _SYMBOLS_CACHE
    if _SYMBOLS_CACHE is not None:
        return _SYMBOLS_CACHE
    r = SESSION.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    info = r.json()
    _SYMBOLS_CACHE = {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}
    return _SYMBOLS_CACHE

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
    """Fetch hourly klines in [start, end)."""
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

# ------------- Resample hourly → noon -------------
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

# ------------- Indicators -------------
def wma(series: pd.Series, window: int):
    s = series.dropna()
    if len(s) < window:
        return np.nan
    weights = np.arange(1, window + 1, dtype=float)
    return float((s.tail(window).to_numpy() * weights).sum() / weights.sum())

def classify_position(close, wma50, wma200):
    if close is None or pd.isna(close) or pd.isna(wma50) or pd.isna(wma200):
        return "Not enough data"
    if close > max(wma50, wma200): return "Above both"
    if close < min(wma50, wma200): return "Below both"
    return "Between"

# ------------- Supabase helpers (date-span aware) -------------
def get_date_stats_from_supabase(coin: str) -> tuple[date | None, date | None]:
    """(min_date, max_date) for coin, or (None,None)."""
    try:
        resp = (
            supabase.table("coin_wma")
            .select("min_date:min(date),max_date:max(date)")
            .eq("coin", coin)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return (None, None)
        md = rows[0].get("min_date")
        xd = rows[0].get("max_date")
        min_d = pd.to_datetime(md).date() if md else None
        max_d = pd.to_datetime(xd).date() if xd else None
        return (min_d, max_d)
    except Exception as e:
        print(f"{coin}: could not read date stats from Supabase: {e}")
        return (None, None)

def dates_to_fetch_for_coin(coin: str, backfill_days: int = MIN_ROWS_FOR_WMA200) -> list[date]:
    """
    Build exact set of local dates needed so the coin has a continuous
    window up to today with at least `backfill_days` coverage.
    """
    today_local = datetime.now(tz=IST).date()
    min_d, max_d = get_date_stats_from_supabase(coin)

    # No data -> last backfill_days
    if max_d is None:
        start = today_local - pd.Timedelta(days=backfill_days - 1)
        return pd.date_range(start, today_local, freq="D").date.tolist()

    need = set()

    # Forward fill (catch up to today)
    if max_d < today_local:
        forward = pd.date_range(max_d + pd.Timedelta(days=1), today_local, freq="D").date.tolist()
        need.update(forward)

    # Ensure at least backfill_days coverage ending today
    span_days = (today_local - (min_d or today_local)).days + 1
    if span_days < backfill_days:
        target_start = today_local - pd.Timedelta(days=backfill_days - 1)
        if min_d is None or min_d > target_start:
            older = pd.date_range(target_start, (min_d - pd.Timedelta(days=1)) if min_d else today_local, freq="D").date.tolist()
            need.update(older)

    return sorted(list(need))

# ------------- Hourly → noon for needed dates -------------
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

# ------------- Main -------------
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

    # Fill to 30 tradable pairs
    bases_extra = get_top_symbols_for_headroom(limit=200)
    all_bases = must_have + [b for b in bases_extra if b not in must_have]
    pairs_all = map_to_valid_pair(all_bases, "USDT")
    missing_must = [b for b in must_have if f"{b}USDT" not in set(pairs_all)]
    if missing_must:
        print("Must-have symbols NOT available on Binance (skipped):", missing_must)
    pairs = pairs_all[:30]
    print(f"✅ Tradable Binance pairs (up to 30, must-haves first): {pairs}")

    all_rows: list[dict] = []

    for pair in pairs:
        sym = pair.replace("USDT", "")
        try:
            need_dates = dates_to_fetch_for_coin(sym, backfill_days=MIN_ROWS_FOR_WMA200)
            if not need_dates:
                print(f"{sym}: up to date; skipping.")
                continue

            daily_new = get_noon_series_for_dates_hourly(sym, need_dates)
            if daily_new is None or daily_new.empty:
                print(f"{sym}: no data for needed dates; skipping.")
                continue

            # Bring prior window (up to 199 days) to compute accurate WMAs at boundary
            try:
                prev_start = (min(need_dates) - pd.Timedelta(days=199)).isoformat()
                resp_prev = (
                    supabase.table("coin_wma")
                    .select("date,close")
                    .eq("coin", sym)
                    .gte("date", prev_start)
                    .order("date")
                    .execute()
                )
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

            # Only new dates are upserted
            mask_new = daily["date"].dt.date.isin(need_dates)
            out = daily.loc[mask_new, ["date","close","WMA_50","WMA_200","Position","Previous_Position"]].copy()
            out["coin"] = sym
            out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
            out = out.rename(columns={"WMA_50":"wma_50","WMA_200":"wma_200","Position":"position","Previous_Position":"previous_position"})

            # JSON-safe
            for col in ["close","wma_50","wma_200"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.replace({np.nan: None})

            rows = out.to_dict(orient="records")
            all_rows.extend(rows)

            # Alert on change if latest day we INSERTED is among need_dates
            if len(daily) >= 2:
                last_day = pd.to_datetime(daily.iloc[-1]["date"]).date()
                if last_day in need_dates:
                    prev, latest = daily.iloc[-2], daily.iloc[-1]
                    if latest["Position"] != prev["Position"]:
                        msg = f"{sym} changed position: {prev['Position']} → {latest['Position']}\nClose: {latest['close']}"
                        send_telegram_alert(msg, coin=sym)

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")

    if not all_rows:
        print("Nothing new to upload — all up to date.")
        return

    # Upsert (dedup within this run)
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
