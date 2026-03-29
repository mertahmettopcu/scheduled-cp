#!/usr/bin/env python3
"""
Binance USD-M perpetual futures market-data pipeline for BTCUSDT and PAXGUSDT.

What it does
------------
- Fetches continuous PERPETUAL klines for 15m / 1h / 1d
- Stores raw candles in Supabase table: futures_candles
- Computes latest strategy snapshot for 15m and 1h
- Computes latest daily Ichimoku snapshot
- Stores latest summary in Supabase table: futures_signal_snapshots

Why this shape
--------------
Keeping raw candles in the database lets the dashboard render charts without
calling Binance Futures directly. That matters because dashboard environments
may get blocked even if GitHub Actions + VPN works.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from supabase import Client, create_client

# ==================== Config ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BINANCE_FUTURES_API = "https://fapi.binance.com"
REQUEST_TIMEOUT = 30
PAIR_LIST = ["BTCUSDT", "PAXGUSDT"]
TIMEFRAME_LIMITS = {
    "15m": 1500,
    "1h": 1200,
    "1d": 400,
}
UPSERT_CHUNK_SIZE = 500
NOON_TZ = "Europe/Istanbul"


def log(msg: str) -> None:
    print(msg, flush=True)


# ==================== Fetch ====================
def fetch_continuous_klines(pair: str, interval: str, limit: int) -> List[List]:
    params = {
        "pair": pair,
        "contractType": "PERPETUAL",
        "interval": interval,
        "limit": limit,
    }

    attempt = 0
    backoff = 0.5
    while True:
        attempt += 1
        try:
            r = requests.get(
                f"{BINANCE_FUTURES_API}/fapi/v1/continuousKlines",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code in (429, 418, 451) or 500 <= r.status_code < 600:
                if attempt < 6:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Binance response for {pair} {interval}: {data}")
            return data
        except Exception as e:
            if attempt < 6:
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            raise RuntimeError(f"Failed to fetch {pair} {interval} after {attempt} attempts: {e}")



def klines_to_df(pair: str, interval: str, raw: List[List]) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df

    df["pair"] = pair
    df["timeframe"] = interval
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)
    return df


# ==================== Indicators ====================
def add_ema_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for length in (4, 16, 65, 120):
        out[f"ema{length}"] = out["close"].ewm(span=length, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    def _rsi(period: int) -> pd.Series:
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    out["rsi14"] = _rsi(14)
    out["rsi52"] = _rsi(52)
    out["ema4_slope"] = out["ema4"].diff()
    out["ema16_slope"] = out["ema16"].diff()

    prev_ema4 = out["ema4"].shift(1)
    prev_ema16 = out["ema16"].shift(1)
    out["long_signal"] = (
        (out["ema4"] > out["ema16"]) &
        (prev_ema4 <= prev_ema16) &
        (out["rsi14"] > out["rsi52"])
    )
    out["short_signal"] = (
        (out["ema4"] < out["ema16"]) &
        (prev_ema4 >= prev_ema16) &
        (out["rsi14"] < out["rsi52"])
    )
    return out



def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high_9 = out["high"].rolling(9).max()
    low_9 = out["low"].rolling(9).min()
    out["tenkan"] = (high_9 + low_9) / 2

    high_26 = out["high"].rolling(26).max()
    low_26 = out["low"].rolling(26).min()
    out["kijun"] = (high_26 + low_26) / 2

    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(26)

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()
    out["senkou_b"] = ((high_52 + low_52) / 2).shift(26)

    out["chikou"] = out["close"].shift(-26)
    return out



def latest_strategy_snapshot(pair: str, timeframe: str, df: pd.DataFrame) -> Dict:
    df_feat = add_ema_rsi_features(df)
    last = df_feat.iloc[-1]

    signal = "NONE"
    if bool(last.get("long_signal", False)):
        signal = "LONG"
    elif bool(last.get("short_signal", False)):
        signal = "SHORT"

    return {
        "pair": pair,
        "timeframe": timeframe,
        "last_open_time": last["open_time"].isoformat().replace("+00:00", "Z"),
        "close": _safe_float(last["close"]),
        "ema4": _safe_float(last["ema4"]),
        "ema16": _safe_float(last["ema16"]),
        "ema65": _safe_float(last["ema65"]),
        "ema120": _safe_float(last["ema120"]),
        "rsi14": _safe_float(last["rsi14"]),
        "rsi52": _safe_float(last["rsi52"]),
        "ema4_slope": _safe_float(last["ema4_slope"]),
        "ema16_slope": _safe_float(last["ema16_slope"]),
        "signal": signal,
        "signal_reason": _signal_reason(last),
        "updated_at": _now_iso(),
    }



def latest_daily_ichimoku_snapshot(pair: str, df: pd.DataFrame) -> Dict:
    daily = add_ichimoku(df)
    last = daily.iloc[-1]

    cloud_top = np.nanmax([last.get("senkou_a", np.nan), last.get("senkou_b", np.nan)])
    cloud_bottom = np.nanmin([last.get("senkou_a", np.nan), last.get("senkou_b", np.nan)])

    if np.isfinite(cloud_top) and last["close"] > cloud_top:
        regime = "ABOVE_CLOUD"
    elif np.isfinite(cloud_bottom) and last["close"] < cloud_bottom:
        regime = "BELOW_CLOUD"
    else:
        regime = "IN_CLOUD"

    return {
        "pair": pair,
        "timeframe": "1d",
        "last_open_time": last["open_time"].isoformat().replace("+00:00", "Z"),
        "close": _safe_float(last["close"]),
        "tenkan": _safe_float(last.get("tenkan")),
        "kijun": _safe_float(last.get("kijun")),
        "senkou_a": _safe_float(last.get("senkou_a")),
        "senkou_b": _safe_float(last.get("senkou_b")),
        "chikou": _safe_float(last.get("chikou")),
        "signal": regime,
        "signal_reason": "Daily Ichimoku regime",
        "updated_at": _now_iso(),
    }


# ==================== Supabase ====================
def candles_to_records(df: pd.DataFrame) -> List[Dict]:
    recs: List[Dict] = []
    for _, row in df.iterrows():
        recs.append({
            "pair": row["pair"],
            "timeframe": row["timeframe"],
            "open_time": row["open_time"].isoformat().replace("+00:00", "Z"),
            "close_time": row["close_time"].isoformat().replace("+00:00", "Z"),
            "open": _safe_float(row["open"]),
            "high": _safe_float(row["high"]),
            "low": _safe_float(row["low"]),
            "close": _safe_float(row["close"]),
            "volume": _safe_float(row["volume"]),
            "quote_asset_volume": _safe_float(row["quote_asset_volume"]),
            "number_of_trades": int(row["number_of_trades"]) if pd.notna(row["number_of_trades"]) else None,
            "taker_buy_base_asset_volume": _safe_float(row["taker_buy_base_asset_volume"]),
            "taker_buy_quote_asset_volume": _safe_float(row["taker_buy_quote_asset_volume"]),
        })
    return recs



def upsert_in_chunks(table_name: str, rows: Iterable[Dict], on_conflict: str) -> None:
    rows = list(rows)
    if not rows:
        return
    for i in range(0, len(rows), UPSERT_CHUNK_SIZE):
        chunk = rows[i:i + UPSERT_CHUNK_SIZE]
        supabase.table(table_name).upsert(chunk, on_conflict=on_conflict, returning="minimal").execute()


# ==================== Utils ====================
def _safe_float(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if pd.isna(v):
        return None
    return float(v)



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")



def _signal_reason(last: pd.Series) -> str:
    if bool(last.get("long_signal", False)):
        return "EMA4 crossed above EMA16 and RSI14 > RSI52"
    if bool(last.get("short_signal", False)):
        return "EMA4 crossed below EMA16 and RSI14 < RSI52"
    return "No fresh crossover signal"


# ==================== Run ====================
def run() -> None:
    candle_rows: List[Dict] = []
    snapshot_rows: List[Dict] = []

    for pair in PAIR_LIST:
        log(f"• Processing {pair}")
        for timeframe, limit in TIMEFRAME_LIMITS.items():
            raw = fetch_continuous_klines(pair, timeframe, limit)
            df = klines_to_df(pair, timeframe, raw)
            if df.empty:
                log(f"  └─ {timeframe}: empty")
                continue

            candle_rows.extend(candles_to_records(df))
            log(f"  └─ {timeframe}: fetched {len(df)} candles")

            if timeframe in ("15m", "1h"):
                snapshot_rows.append(latest_strategy_snapshot(pair, timeframe, df))
            elif timeframe == "1d":
                snapshot_rows.append(latest_daily_ichimoku_snapshot(pair, df))

    if not candle_rows:
        raise SystemExit("No candle rows fetched; nothing to upsert.")

    # Deduplicate in-memory by (pair, timeframe, open_time)
    dedup_candles = {(r["pair"], r["timeframe"], r["open_time"]): r for r in candle_rows}
    dedup_snapshots = {(r["pair"], r["timeframe"]): r for r in snapshot_rows}

    log(f"⬆️ Upserting {len(dedup_candles)} candle rows into futures_candles")
    upsert_in_chunks("futures_candles", dedup_candles.values(), on_conflict="pair,timeframe,open_time")

    log(f"⬆️ Upserting {len(dedup_snapshots)} snapshot rows into futures_signal_snapshots")
    upsert_in_chunks("futures_signal_snapshots", dedup_snapshots.values(), on_conflict="pair,timeframe")

    log("✅ Pipeline complete")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
