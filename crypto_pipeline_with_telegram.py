#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py

- Ensure a must-have coin list is attempted first
- Fetch extra coins from CoinGecko to fill to 30 total
- Use Binance 1-minute klines to capture the EXACT 12:00 Europe/Istanbul close per day
- Compute WMA(50) and WMA(200) with strict windows
- Classify position and previous_position
- Upsert into Supabase (coin_wma table) with (coin,date) conflict target
- Send Telegram alerts on position change (latest vs previous)

Env:
  SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Run:
  python crypto_pipeline_with_telegram.py
"""

import os
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client, Client
from zoneinfo import ZoneInfo


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

def fetch_minute_klines_exact(pair: str, start_ms: int, limit: int = 1):
    """
    Fetch 1-minute klines starting at start_ms (UTC ms).
    With limit=1, Binance returns the candle whose open >= start_ms.
    We validate the exact open_time (== start_ms) afterwards.
    """
    params = {
        "symbol": pair,
        "interval": "1m",
        "startTime": start_ms,
        "limit": limit,
    }
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# -------------------------- Time alignment ---------------------------------------------
def get_binance_noon_series(symbol: str, days: int = 250):
    """
    Build a daily series using the exact 12:00 (Europe/Istanbul) close for each day
    by fetching the 1-minute kline that starts at exactly that minute.
    Returns DataFrame with columns ['date','close'] sorted ascending by date.
    """
    pair = f"{symbol}USDT"

    # Build the set of local dates we want (unique, sorted), using zoneinfo to avoid pandas tz issues
    now_utc = datetime.now(timezone.utc)
    ist = ZoneInfo("Europe/Istanbul")

    dates_local = []
    # +10 extra days as margin so WMA rolling windows have breathing room
    for d in range(days + 10):
        # Take "today in Istanbul", step back d days, and keep DATE only
        ist_now = now_utc.astimezone(ist)
        target_local_date = (ist_now.date() - pd.Timedelta(days=d).to_pytimedelta())
        dates_local.append(target_local_date)
    dates_local = sorted(set(dates_local))

    rows = []
    for d in dates_local:
        # Build 12:00 local time (tz-aware), then convert to UTC ms
        ts_local = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=ist)
        ts_utc = ts_local.astimezone(timezone.utc)
        start_ms = int(ts_utc.timestamp() * 1000)

        try:
            # First try: exact 12:00 minute
            data = fetch_minute_klines_exact(pair, start_ms=start_ms, limit=1)
            got_exact = False
            if data:
                open_ms = data[0][0]
                if open_ms == start_ms:
                    close_price = float(data[0][4])
                    rows.append({"date": pd.Timestamp(d), "close": close_price})
                    got_exact = True

            if not got_exact:
                # Fallback: small window around 12:00 (11:59–12:01) to find the exact opening minute
                window_start = start_ms - 60_000
                data2 = fetch_minute_klines_exact(pair, start_ms=window_start, limit=3)
                found = False
                for k in data2 or []:
                    if k[0] == start_ms:
                        close_price = float(k[4])
                        rows.append({"date": pd.Timestamp(d), "close": close_price})
                        found = True
                        break
                if not found:
                    print(f"[{pair}] Missing exact 12:00 local minute for {d}")
            time.sleep(0.05)  # be gentle with API

        except requests.HTTPError as e:
            print(f"[{pair}] HTTP error for {d}: {e}")
            time.sleep(0.2)
        except Exception as e:
            print(f"[{pair}] Error for {d}: {e}")
            time.sleep(0.2)

    if not rows:
        return None

    daily = pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)
    return daily


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
def run_pipeline(days=250):
    # 1) Must-have bases (normalized)
    must_have_raw = [
        "BTC","ETH","XRP","BNB","SOL","DOGE","TRX","ADA",
        "HYPE","XLM","SUI","LINK","BCH","HBAR","AVAX","TON",
        "LTC","1000 SHIB","DOT","UNIS"
    ]
    normalize = {
        "1000SHIB": "SHIB",
        "UNIS": "UNI",
    }
    def norm(sym: str) -> str:
        s = sym.upper().replace(" ", "")
        return normalize.get(s, s)

    must_have = [norm(s) for s in must_have_raw]  # e.g., "1000SHIB" -> "SHIB", "UNIS" -> "UNI"

    # 2) Add extra bases from CoinGecko to fill to 30
    bases_extra = get_top_symbols_for_headroom(limit=200)
    all_bases = must_have + [b for b in bases_extra if b not in must_have]

    # 3) Map to Binance tradable USDT pairs
    pairs_all = map_to_valid_pair(all_bases, "USDT")

    # Log any must-haves that did not resolve to a Binance pair
    listed_pairs_set = set(pairs_all)
    missing_must = [b for b in must_have if f"{b}USDT" not in listed_pairs_set]
    if missing_must:
        print("Must-have symbols NOT available on Binance (skipped):", missing_must)

    # Keep exactly the first 30 tradable pairs (must-haves come first)
    pairs = pairs_all[:30]
    print(f"Tradable Binance pairs (up to 30, must-haves first): {pairs}")

    # 4) Process & upsert
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

            # Strict windows to avoid partial indicators
            daily["WMA_50"] = daily["close"].rolling(window=50, min_periods=50)\
                                            .apply(lambda s: wma(s, 50), raw=False)
            daily["WMA_200"] = daily["close"].rolling(window=200, min_periods=200)\
                                             .apply(lambda s: wma(s, 200), raw=False)

            daily["Position"] = daily.apply(
                lambda r: classify_position(r["close"], r["WMA_50"], r["WMA_200"]),
                axis=1
            )
            daily["Previous_Position"] = daily["Position"].shift(1)

            # Only keep rows where both WMAs exist (strict)
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
