#!/usr/bin/env python3
"""
Binance USD-M perpetual futures market-data pipeline.

Updates in this version
-----------------------
- Reads symbol list from Google Sheets CSV (symbol, enabled)
- Supports separate Telegram signal chats and alert chats
- Waits for the next 15m candle close when scheduled early
- Drops still-open candles so signals use only closed candles
- Sends safer Telegram diagnostics instead of crashing on one bad chat
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from supabase import Client, create_client

# ==================== Config ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "")
DISPLAY_TZ = "Europe/Istanbul"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_SIGNAL_CHAT_IDS_RAW = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_ALERT_CHAT_IDS_RAW = os.getenv("TELEGRAM_ALERT_CHAT_IDS", "")
TELEGRAM_SIGNAL_CHAT_IDS = [x.strip() for x in TELEGRAM_SIGNAL_CHAT_IDS_RAW.split(",") if x.strip()]
TELEGRAM_ALERT_CHAT_IDS = [x.strip() for x in TELEGRAM_ALERT_CHAT_IDS_RAW.split(",") if x.strip()]

SYMBOLS_SHEET_CSV_URL = os.getenv("GOOGLE_SHEETS_SYMBOLS_CSV_URL", "").strip()
DEFAULT_PAIR_LIST = ["BTCUSDT", "PAXGUSDT"]

BINANCE_FUTURES_API = "https://fapi.binance.com"
REQUEST_TIMEOUT = 30
TIMEFRAME_LIMITS = {
    "15m": 1500,
    "1h": 1200,
    "1d": 400,
}
UPSERT_CHUNK_SIZE = 500

WAIT_FOR_CANDLE_CLOSE = os.getenv("WAIT_FOR_CANDLE_CLOSE", "0").strip().lower() in {"1", "true", "yes", "y"}
MAX_CLOSE_WAIT_SECONDS = int(os.getenv("MAX_CLOSE_WAIT_SECONDS", "480"))
CLOSE_BUFFER_SECONDS = int(os.getenv("CLOSE_BUFFER_SECONDS", "5"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def log(msg: str) -> None:
    print(msg, flush=True)


# ==================== Time / scheduling ====================
def compute_next_15m_close(now_utc: datetime) -> datetime:
    floored_minute = (now_utc.minute // 15) * 15
    floor = now_utc.replace(minute=floored_minute, second=0, microsecond=0)
    if floor == now_utc:
        return floor
    return floor + timedelta(minutes=15)


def get_run_cutoff_utc() -> datetime:
    """
    When WAIT_FOR_CANDLE_CLOSE is enabled, wait until just after the next 15m close,
    then use that close moment as the cutoff. This guarantees only closed candles are used.
    """
    now_utc = datetime.now(timezone.utc)

    if not WAIT_FOR_CANDLE_CLOSE:
        return now_utc

    target_close = compute_next_15m_close(now_utc)
    seconds_to_close = (target_close - now_utc).total_seconds()

    if seconds_to_close < 0:
        return now_utc

    if seconds_to_close > MAX_CLOSE_WAIT_SECONDS:
        raise RuntimeError(
            f"Scheduled close-wait window is too early ({seconds_to_close:.0f}s before close). "
            f"Check workflow cron vs WAIT_FOR_CANDLE_CLOSE settings."
        )

    wait_seconds = seconds_to_close + CLOSE_BUFFER_SECONDS
    log(
        f"⏳ Waiting {wait_seconds:.0f}s for next 15m candle close "
        f"at {target_close.isoformat().replace('+00:00', 'Z')}"
    )
    time.sleep(wait_seconds)
    return target_close


def filter_closed_candles(df: pd.DataFrame, cutoff_utc: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff_ts = pd.Timestamp(cutoff_utc)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize("UTC")
    filtered = df[df["close_time"] <= cutoff_ts].copy()
    return filtered.reset_index(drop=True)


# ==================== Symbols from Google Sheets ====================
def _parse_enabled(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def load_pairs_from_sheet() -> List[str]:
    if not SYMBOLS_SHEET_CSV_URL:
        log("⚠️ GOOGLE_SHEETS_SYMBOLS_CSV_URL not set, falling back to default pair list")
        return DEFAULT_PAIR_LIST

    try:
        df = pd.read_csv(SYMBOLS_SHEET_CSV_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to read symbols sheet CSV: {e}")

    if df.empty:
        raise RuntimeError("Symbols sheet is empty")

    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"symbol", "enabled"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError("Symbols sheet must contain columns: symbol, enabled")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"] != ""]
    df = df[df["symbol"].str.match(r"^[A-Z0-9_\-]+$")]
    df["enabled_bool"] = df["enabled"].apply(_parse_enabled)

    pairs = df.loc[df["enabled_bool"], "symbol"].drop_duplicates().tolist()
    if not pairs:
        raise RuntimeError("Symbols sheet has no enabled symbols")

    log(f"📄 Loaded {len(pairs)} enabled symbols from Google Sheets: {', '.join(pairs)}")
    return pairs


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
        return 100 - (100 / (1 + rs))

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

    out["senkou_a_base"] = (out["tenkan"] + out["kijun"]) / 2
    out["senkou_a"] = out["senkou_a_base"].shift(26)

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()
    out["senkou_b_base"] = (high_52 + low_52) / 2
    out["senkou_b"] = out["senkou_b_base"].shift(26)

    out["chikou"] = out["close"].shift(-26)
    return out


# ==================== Telegram ====================
def _telegram_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def _send_telegram_text(text: str, chat_ids: List[str], label: str) -> List[str]:
    if not TELEGRAM_BOT_TOKEN or not chat_ids:
        log(f"Telegram {label} credentials missing, skipping send.")
        return []

    errors: List[str] = []
    url = _telegram_api_url("sendMessage")

    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": text}
        try:
            r = requests.post(url, json=payload, timeout=20)
            if not r.ok:
                try:
                    details = r.json()
                except Exception:
                    details = r.text
                err = f"chat_id={chat_id} status={r.status_code} details={details}"
                errors.append(err)
                log(f"Telegram {label} failed: {err}")
                continue
            log(f"Telegram {label} sent to chat_id={chat_id}")
        except Exception as e:
            err = f"chat_id={chat_id} exception={e}"
            errors.append(err)
            log(f"Telegram {label} failed: {err}")

    return errors


def send_telegram_message(text: str) -> None:
    errors = _send_telegram_text(text, TELEGRAM_SIGNAL_CHAT_IDS, label="signal")
    if errors:
        send_alert_message("⚠️ Signal Telegram send had errors:\n" + "\n".join(errors[:10]))


def send_alert_message(text: str) -> None:
    _send_telegram_text(text, TELEGRAM_ALERT_CHAT_IDS, label="alert")


# ==================== Classification ====================
def classify_ema_rsi_signal(row: pd.Series) -> str:
    ema_bull = row["ema4"] > row["ema16"]
    ema_bear = row["ema4"] < row["ema16"]
    rsi_bull = row["rsi14"] > row["rsi52"]
    rsi_bear = row["rsi14"] < row["rsi52"]

    if ema_bull and rsi_bull:
        return "LONG"
    if ema_bear and rsi_bear:
        return "SHORT"
    return "NEUTRAL"


def lagging_span_position(df: pd.DataFrame) -> str:
    if len(df) < 27:
        return "unknown"

    current_close = df.iloc[-1]["close"]
    ref_candle = df.iloc[-27]
    ref_high = ref_candle["high"]
    ref_low = ref_candle["low"]

    if current_close > ref_high:
        return "above candle range"
    if current_close < ref_low:
        return "below candle range"
    return "inside candle range"


def current_cloud_position(row: pd.Series) -> str:
    a = row["senkou_a"]
    b = row["senkou_b"]

    if pd.isna(a) or pd.isna(b):
        return "unknown"

    cloud_top = max(a, b)
    cloud_bottom = min(a, b)
    close_ = row["close"]

    if close_ > cloud_top:
        return "strictly above current cloud"
    if close_ < cloud_bottom:
        return "strictly below current cloud"
    return "inside current cloud"


def future_cloud_color(row: pd.Series) -> str:
    a = row.get("senkou_a_base")
    b = row.get("senkou_b_base")

    if pd.isna(a) or pd.isna(b):
        return "unknown"

    if a > b:
        return "green"
    if a < b:
        return "red"
    return "flat"


def classify_ichimoku_signal(df: pd.DataFrame) -> tuple[str, dict]:
    row = df.iloc[-1]

    conv_gt_base = row["tenkan"] > row["kijun"]
    conv_lt_base = row["tenkan"] < row["kijun"]

    lag_pos = lagging_span_position(df)
    cloud_pos = current_cloud_position(row)
    future_color = future_cloud_color(row)

    bullish = (
        lag_pos == "above candle range"
        and conv_gt_base
        and cloud_pos == "strictly above current cloud"
        and future_color == "green"
    )

    bearish = (
        lag_pos == "below candle range"
        and conv_lt_base
        and cloud_pos == "strictly below current cloud"
        and future_color == "red"
    )

    if bullish:
        signal = "LONG"
    elif bearish:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    details = {
        "lagging_span": lag_pos,
        "conversion_vs_base": (
            "Conversion line > Base line" if conv_gt_base
            else "Conversion line < Base line" if conv_lt_base
            else "Conversion line = Base line"
        ),
        "cloud_position": f"Close: {cloud_pos}",
        "current_cloud": (
            "Current cloud: green" if row["senkou_a"] > row["senkou_b"]
            else "Current cloud: red" if row["senkou_a"] < row["senkou_b"]
            else "Current cloud: flat"
        ),
        "future_cloud": f"Future cloud: {future_color}",
    }
    return signal, details


def build_telegram_message(
    pair: str,
    row_15m: pd.Series,
    signal_15m: str,
    row_1h: pd.Series,
    signal_1h: str,
    row_1d: pd.Series,
    signal_1d: str,
    ichi_details: dict,
    candle_time_15m: str,
    candle_time_1h: str,
    candle_time_1d: str,
    triggered_15m: bool = False,
    triggered_1h: bool = False,
    triggered_1d: bool = False,
) -> str:
    def fmt(x):
        if pd.isna(x):
            return "NA"
        return f"{x:.4f}"

    def sig_label(sig: str, triggered: bool) -> str:
        return f"{sig} (triggered)" if triggered else sig

    app_link = _build_app_link(pair)

    lines = [
        f"{pair}",
        f"15m candle: {_format_candle_time_for_message(candle_time_15m)}",
        f"1h candle: {_format_candle_time_for_message(candle_time_1h)}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    if app_link:
        lines.append(f"Chart: {app_link}")

    lines.extend([
        "",
        f"15m: {sig_label(signal_15m, triggered_15m)}",
        f"Close: {fmt(row_15m['close'])}",
        "EMA4 > EMA16" if row_15m["ema4"] > row_15m["ema16"] else "EMA4 < EMA16" if row_15m["ema4"] < row_15m["ema16"] else "EMA4 = EMA16",
        "RSI14 > RSI52" if row_15m["rsi14"] > row_15m["rsi52"] else "RSI14 < RSI52" if row_15m["rsi14"] < row_15m["rsi52"] else "RSI14 = RSI52",
        "",
        f"1h: {sig_label(signal_1h, triggered_1h)}",
        f"Close: {fmt(row_1h['close'])}",
        "EMA4 > EMA16" if row_1h["ema4"] > row_1h["ema16"] else "EMA4 < EMA16" if row_1h["ema4"] < row_1h["ema16"] else "EMA4 = EMA16",
        "RSI14 > RSI52" if row_1h["rsi14"] > row_1h["rsi52"] else "RSI14 < RSI52" if row_1h["rsi14"] < row_1h["rsi52"] else "RSI14 = RSI52",
        "",
        f"1d: {sig_label(signal_1d, triggered_1d)}",
        f"Close: {fmt(row_1d['close'])}",
        f"Lagging span: {ichi_details['lagging_span']}",
        ichi_details["conversion_vs_base"],
        ichi_details["cloud_position"],
        ichi_details.get("current_cloud", "Current cloud: unknown"),
        ichi_details["future_cloud"],
    ])
    return "\n".join(lines)


def signal_changed(prev_value: str | None, new_value: str) -> bool:
    prev_norm = (prev_value or "").strip().upper()
    new_norm = (new_value or "").strip().upper()
    return prev_norm != new_norm


# ==================== Snapshots ====================
def latest_strategy_snapshot(pair: str, timeframe: str, df: pd.DataFrame) -> Dict:
    df_feat = add_ema_rsi_features(df)
    last = df_feat.iloc[-1]

    signal = classify_ema_rsi_signal(last)

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

    signal, ichi_details = classify_ichimoku_signal(daily)

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
        "signal": signal,
        "signal_reason": "Daily Ichimoku regime",
        "ichimoku_lagging_span": ichi_details["lagging_span"],
        "ichimoku_conversion_vs_base": ichi_details["conversion_vs_base"],
        "ichimoku_cloud_position": ichi_details["cloud_position"],
        "ichimoku_future_cloud": ichi_details["future_cloud"],
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


def get_previous_snapshot_map(pair: str) -> Dict[str, Dict]:
    resp = (
        supabase.table("futures_signal_snapshots")
        .select("*")
        .eq("pair", pair)
        .execute()
    )
    rows = resp.data or []
    return {row["timeframe"]: row for row in rows}


# ==================== Utils ====================
def _safe_float(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if pd.isna(v):
        return None
    return float(v)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_candle_time_for_message(iso_ts: str) -> str:
    ts = pd.Timestamp(iso_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(DISPLAY_TZ)
    return ts.strftime("%Y-%m-%d %H:%M")


def _build_app_link(pair: str) -> str:
    if not STREAMLIT_APP_URL:
        return ""
    separator = "&" if "?" in STREAMLIT_APP_URL else "?"
    return f"{STREAMLIT_APP_URL}{separator}pair={pair}"


def _signal_reason(last: pd.Series) -> str:
    if bool(last.get("long_signal", False)):
        return "EMA4 crossed above EMA16 and RSI14 > RSI52"
    if bool(last.get("short_signal", False)):
        return "EMA4 crossed below EMA16 and RSI14 < RSI52"
    return "No fresh crossover signal"


# ==================== Run ====================
def run() -> None:
    run_cutoff_utc = get_run_cutoff_utc()
    pair_list = load_pairs_from_sheet()

    candle_rows: List[Dict] = []
    snapshot_rows: List[Dict] = []

    for pair in pair_list:
        log(f"• Processing {pair}")

        pair_dfs: Dict[str, pd.DataFrame] = {}

        for timeframe, limit in TIMEFRAME_LIMITS.items():
            raw = fetch_continuous_klines(pair, timeframe, limit)
            df = klines_to_df(pair, timeframe, raw)
            df = filter_closed_candles(df, run_cutoff_utc)

            if df.empty:
                log(f"  └─ {timeframe}: empty after closed-candle filter")
                continue

            pair_dfs[timeframe] = df
            candle_rows.extend(candles_to_records(df))
            log(f"  └─ {timeframe}: fetched {len(df)} closed candles")

        if "15m" not in pair_dfs or "1h" not in pair_dfs or "1d" not in pair_dfs:
            log(f"  └─ Skipping {pair} snapshot generation because one or more timeframes are missing")
            continue

        snap_15m = latest_strategy_snapshot(pair, "15m", pair_dfs["15m"])
        snap_1h = latest_strategy_snapshot(pair, "1h", pair_dfs["1h"])
        snap_1d = latest_daily_ichimoku_snapshot(pair, pair_dfs["1d"])

        snapshot_rows.extend([snap_15m, snap_1h, snap_1d])

        prev_map = get_previous_snapshot_map(pair)
        prev_15m = prev_map.get("15m")
        prev_1h = prev_map.get("1h")
        prev_1d = prev_map.get("1d")

        changed_15m = signal_changed(prev_15m.get("signal") if prev_15m else None, snap_15m["signal"])
        changed_1h = signal_changed(prev_1h.get("signal") if prev_1h else None, snap_1h["signal"])
        changed_1d = signal_changed(prev_1d.get("signal") if prev_1d else None, snap_1d["signal"])

        should_send = (prev_15m is not None or prev_1h is not None or prev_1d is not None) and (
            changed_15m or changed_1h or changed_1d
        )

        if should_send:
            df_15m_feat = add_ema_rsi_features(pair_dfs["15m"])
            df_1h_feat = add_ema_rsi_features(pair_dfs["1h"])
            df_1d_ichi = add_ichimoku(pair_dfs["1d"])

            latest_15m = df_15m_feat.iloc[-1]
            latest_1h = df_1h_feat.iloc[-1]
            latest_1d = df_1d_ichi.iloc[-1]

            ichi_details = {
                "lagging_span": snap_1d.get("ichimoku_lagging_span", "unknown"),
                "conversion_vs_base": snap_1d.get("ichimoku_conversion_vs_base", "unknown"),
                "cloud_position": snap_1d.get("ichimoku_cloud_position", "unknown"),
                "future_cloud": snap_1d.get("ichimoku_future_cloud", "unknown"),
                "current_cloud": (
                    "Current cloud: green" if latest_1d.get("senkou_a") > latest_1d.get("senkou_b")
                    else "Current cloud: red" if latest_1d.get("senkou_a") < latest_1d.get("senkou_b")
                    else "Current cloud: flat"
                ) if pd.notna(latest_1d.get("senkou_a")) and pd.notna(latest_1d.get("senkou_b")) else "Current cloud: unknown",
            }

            message = build_telegram_message(
                pair=pair,
                row_15m=latest_15m,
                signal_15m=snap_15m["signal"],
                row_1h=latest_1h,
                signal_1h=snap_1h["signal"],
                row_1d=latest_1d,
                signal_1d=snap_1d["signal"],
                ichi_details=ichi_details,
                candle_time_15m=snap_15m["last_open_time"],
                candle_time_1h=snap_1h["last_open_time"],
                candle_time_1d=snap_1d["last_open_time"],
                triggered_15m=changed_15m,
                triggered_1h=changed_1h,
                triggered_1d=changed_1d,
            )
            send_telegram_message(message)
            log(f"  └─ 📨 Telegram sent for {pair}")
        else:
            log(f"  └─ Telegram skipped for {pair} (no new signal)")

    if not candle_rows:
        raise SystemExit("No candle rows fetched; nothing to upsert.")

    dedup_candles = {(r["pair"], r["timeframe"], r["open_time"]): r for r in candle_rows}
    dedup_snapshots = {(r["pair"], r["timeframe"]): r for r in snapshot_rows}

    log(f"⬆️ Upserting {len(dedup_candles)} candle rows into futures_candles")
    upsert_in_chunks("futures_candles", dedup_candles.values(), on_conflict="pair,timeframe,open_time")

    log(f"⬆️ Upserting {len(dedup_snapshots)} snapshot rows into futures_signal_snapshots")
    upsert_in_chunks("futures_signal_snapshots", dedup_snapshots.values(), on_conflict="pair,timeframe")

    log("✅ Pipeline complete")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        run()
    except Exception as exc:
        alert_text = (
            "🚨 Crypto pipeline runtime error\n"
            f"Time: {_now_iso()}\n"
            f"Error: {type(exc).__name__}: {exc}"
        )
        send_alert_message(alert_text)
        raise
