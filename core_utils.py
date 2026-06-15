#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import httpx
import numpy as np
import pandas as pd
import requests
from supabase import Client, create_client

DISPLAY_TZ = "Europe/Istanbul"
BINANCE_FUTURES_API = "https://fapi.binance.com"
REQUEST_TIMEOUT = 30
PAIR_LIST = ["BTCUSDT"]
TIMEFRAME_LIMITS = {
    "15m": 1500,
    "1h": 1200,
    "1d": 400,
}
UPSERT_CHUNK_SIZE = 500

SUPABASE_MAX_RETRIES = 4
SUPABASE_RETRY_BASE_DELAY = 1.0
SUPABASE_RETRY_MAX_DELAY = 8.0


def log(msg: str) -> None:
    print(msg, flush=True)


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

def ema_with_sma_seed(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    result = pd.Series(index=series.index, dtype="float64")

    if len(series) < length:
        return result

    first_ema_position = length - 1
    result.iloc[first_ema_position] = series.iloc[:length].mean()

    for i in range(length, len(series)):
        result.iloc[i] = (
            alpha * series.iloc[i]
            + (1 - alpha) * result.iloc[i - 1]
        )

    return result
    
def add_ema_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "open_time" in out.columns:
        out = out.sort_values("open_time").reset_index(drop=True)

    for length in (4, 16, 65, 120, 168):
        out[f"ema{length}"] = out["close"].ewm(span=length, adjust=False).mean()
        
    for length in (4, 16, 65, 120, 168):
        out[f"sma{length}"] = out["close"].rolling(length).mean()

    # Temporary EMA168 diagnostics.
    # Bunlar strateji sinyalini değiştirmez; sadece Binance ile hangi hesap eşleşiyor görmek için.
    out["ema168_adjust_false"] = out["close"].ewm(span=168, adjust=False).mean()
    out["ema168_adjust_true"] = out["close"].ewm(span=168, adjust=True).mean()
    out["sma168"] = out["close"].rolling(168).mean()
    out["ema168_sma_seed"] = ema_with_sma_seed(out["close"], 168)

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

    prev_sma4 = out["sma4"].shift(1)
    prev_sma16 = out["sma16"].shift(1)
    
    out["long_signal"] = (
        (out["sma4"] > out["sma16"]) &
        (prev_sma4 <= prev_sma16) &
        (out["rsi14"] > out["rsi52"]) &
        (out["rsi14"] >= 50)
    )
    
    out["short_signal"] = (
        (out["sma4"] < out["sma16"]) &
        (prev_sma4 >= prev_sma16) &
        (out["rsi14"] < out["rsi52"]) &
        (out["rsi14"] <= 50)
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

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()

    out["senkou_a_base"] = (out["tenkan"] + out["kijun"]) / 2
    out["senkou_b_base"] = (high_52 + low_52) / 2

    out["senkou_a"] = out["senkou_a_base"].shift(26)
    out["senkou_b"] = out["senkou_b_base"].shift(26)
    out["chikou"] = out["close"].shift(-26)
    return out


def classify_ema_rsi_signal(row: pd.Series) -> str:
    # Legacy function name; official pipeline signal now uses SMA4/SMA16.
    sma_bull = row["sma4"] > row["sma16"]
    sma_bear = row["sma4"] < row["sma16"]
    rsi_bull = row["rsi14"] > row["rsi52"]
    rsi_bear = row["rsi14"] < row["rsi52"]

    if sma_bull and rsi_bull and row["rsi14"] >= 50:
        return "LONG"

    if sma_bear and rsi_bear and row["rsi14"] <= 50:
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
    
def format_price(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def signal_badge(sig: str) -> str:
    sig = (sig or "").strip().upper()
    if sig == "LONG":
        return "🟢 LONG"
    if sig == "SHORT":
        return "🔴 SHORT"
    return "⚪ NEUTRAL"


def signal_display(current: str, previous: str | None, triggered: bool, suffix: str = "🔔") -> str:
    current_text = signal_badge(current)
    previous_norm = (previous or "").strip().upper()
    current_norm = (current or "").strip().upper()

    if triggered and previous_norm and previous_norm != current_norm:
        return f"{signal_badge(previous_norm)} → {current_text} {suffix}"

    return current_text


def normalize_signal(sig: str | None) -> str:
    sig = (sig or "").strip().upper()
    if sig in {"LONG", "SHORT", "NEUTRAL"}:
        return sig
    return "NEUTRAL"


def read_ichimoku_tp_multiplier(default: float = 1.7) -> float:
    raw = os.getenv("ICHIMOKU_TP_MULTIPLIER")

    if raw is None or str(raw).strip() == "":
        return float(default)

    try:
        value = float(str(raw).strip())
    except ValueError:
        log(f"WARNING: Invalid ICHIMOKU_TP_MULTIPLIER={raw!r}. Falling back to {default}.")
        return float(default)

    if value <= 0:
        log(f"WARNING: ICHIMOKU_TP_MULTIPLIER must be positive. Falling back to {default}.")
        return float(default)

    return value


def read_counter_momentum_thresholds(
    early_default: float = 20.0,
    full_default: float = 35.0,
) -> tuple[float, float]:
    early_raw = os.getenv("COUNTER_MOMENTUM_EARLY_THRESHOLD")
    full_raw = os.getenv("COUNTER_MOMENTUM_FULL_THRESHOLD")

    def _read_pct(raw, default, name):
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            value = float(str(raw).strip())
        except ValueError:
            log(f"WARNING: Invalid {name}={raw!r}. Falling back to {default}.")
            return float(default)
        if value <= 0:
            log(f"WARNING: {name} must be positive. Falling back to {default}.")
            return float(default)
        return value

    early = _read_pct(early_raw, early_default, "COUNTER_MOMENTUM_EARLY_THRESHOLD")
    full = _read_pct(full_raw, full_default, "COUNTER_MOMENTUM_FULL_THRESHOLD")

    if early > full:
        log(
            "WARNING: COUNTER_MOMENTUM_EARLY_THRESHOLD is greater than "
            "COUNTER_MOMENTUM_FULL_THRESHOLD. Swapping values."
        )
        early, full = full, early

    return early, full


def read_counter_momentum_repeat_mode(default: str = "always") -> str:
    raw = os.getenv("COUNTER_MOMENTUM_REPEAT_MODE", default)
    mode = str(raw or default).strip().lower()

    if mode not in {"always", "new_high_only"}:
        log(
            f"WARNING: Invalid COUNTER_MOMENTUM_REPEAT_MODE={raw!r}. "
            f"Falling back to {default!r}."
        )
        return default

    return mode


def _manual_zone_values(zones: pd.DataFrame) -> list[float]:
    if zones is None or zones.empty or "zone_value" not in zones.columns:
        return []

    values = (
        pd.to_numeric(zones["zone_value"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )
    return [float(v) for v in values]


def find_opening_zone_context(open_price: float, zones: pd.DataFrame) -> Dict:
    zone_values = _manual_zone_values(zones)

    if not zone_values or open_price is None or pd.isna(open_price):
        return {
            "upper_zone": None,
            "lower_zone": None,
            "zone_distance": None,
            "reason": "zone context unavailable",
        }

    open_price = float(open_price)
    upper_candidates = [z for z in zone_values if z > open_price]
    lower_candidates = [z for z in zone_values if z < open_price]

    upper_zone = min(upper_candidates) if upper_candidates else None
    lower_zone = max(lower_candidates) if lower_candidates else None

    if upper_zone is None or lower_zone is None:
        return {
            "upper_zone": upper_zone,
            "lower_zone": lower_zone,
            "zone_distance": None,
            "reason": "open price is outside available zone range",
        }

    zone_distance = float(upper_zone) - float(lower_zone)

    if zone_distance <= 0:
        return {
            "upper_zone": upper_zone,
            "lower_zone": lower_zone,
            "zone_distance": None,
            "reason": "invalid zone distance",
        }

    return {
        "upper_zone": float(upper_zone),
        "lower_zone": float(lower_zone),
        "zone_distance": float(zone_distance),
        "reason": "ok",
    }


def evaluate_opposite_sma_rsi_conditions(
    df_1h_with_current_open: pd.DataFrame,
    reference_signal: str,
) -> Dict:
    reference_signal = normalize_signal(reference_signal)

    if df_1h_with_current_open.empty:
        return {"opposite_signal": "NEUTRAL", "conditions_met": False, "conditions": {}}

    feat = add_ema_rsi_features(df_1h_with_current_open)
    last = feat.iloc[-1]

    if reference_signal == "LONG":
        opposite_signal = "SHORT"
        conditions = {
            "SMA4 < SMA16": bool(last["sma4"] < last["sma16"]),
            "RSI14 < RSI52": bool(last["rsi14"] < last["rsi52"]),
            "RSI14 <= 50": bool(last["rsi14"] <= 50),
        }
    elif reference_signal == "SHORT":
        opposite_signal = "LONG"
        conditions = {
            "SMA4 > SMA16": bool(last["sma4"] > last["sma16"]),
            "RSI14 > RSI52": bool(last["rsi14"] > last["rsi52"]),
            "RSI14 >= 50": bool(last["rsi14"] >= 50),
        }
    else:
        return {"opposite_signal": "NEUTRAL", "conditions_met": False, "conditions": {}}

    return {
        "opposite_signal": opposite_signal,
        "conditions_met": all(conditions.values()),
        "conditions": conditions,
        "sma4": _safe_float(last.get("sma4")),
        "sma16": _safe_float(last.get("sma16")),
        "rsi14": _safe_float(last.get("rsi14")),
        "rsi52": _safe_float(last.get("rsi52")),
    }


def get_last_directional_sma_rsi_reference(df: pd.DataFrame) -> Dict:
    """
    Find the latest closed 1H state-change event where SMA/RSI state became
    LONG or SHORT. NEUTRAL states do not erase the last directional reference.

    This is used as the reference direction for counter momentum warnings.
    """
    if df.empty:
        return {
            "signal": "NEUTRAL",
            "signal_time": None,
            "row": None,
            "reason": "empty dataframe",
        }

    feat = add_ema_rsi_features(df)
    signals = feat.apply(classify_ema_rsi_signal, axis=1)
    prev_signals = signals.shift(1).fillna("NEUTRAL")

    event_mask = (
        signals.isin(["LONG", "SHORT"])
        & (signals != prev_signals)
    )

    events = feat.loc[event_mask].copy()
    events["reference_signal"] = signals.loc[event_mask].values

    if events.empty:
        return {
            "signal": "NEUTRAL",
            "signal_time": None,
            "row": None,
            "reason": "no directional state-change event",
        }

    last_event = events.iloc[-1]
    signal = str(last_event["reference_signal"]).upper()
    signal_time = last_event["open_time"].isoformat().replace("+00:00", "Z")

    return {
        "signal": signal,
        "signal_time": signal_time,
        "row": last_event,
        "reason": "ok",
    }


def evaluate_counter_momentum(
    *,
    pair: str,
    reference_signal: str | None,
    reference_signal_time: str | None,
    open_1h_row: pd.Series,
    df_1h_with_current_open: pd.DataFrame,
    zones: pd.DataFrame,
    early_threshold_pct: float,
    full_threshold_pct: float,
) -> Dict:
    reference_signal = normalize_signal(reference_signal)

    if reference_signal not in {"LONG", "SHORT"}:
        return {
            "should_warn": False,
            "reason": "no directional reference signal",
            "reference_signal": reference_signal,
        }

    candle_open = float(open_1h_row["open"])
    current_price = float(open_1h_row["close"])
    body_size = abs(current_price - candle_open)

    if current_price == candle_open:
        return {
            "should_warn": False,
            "reason": "open candle body is zero",
            "reference_signal": reference_signal,
            "candle_open": candle_open,
            "current_price": current_price,
            "body_size": 0.0,
        }

    candle_direction = "LONG" if current_price > candle_open else "SHORT"
    expected_counter_direction = "SHORT" if reference_signal == "LONG" else "LONG"

    if candle_direction != expected_counter_direction:
        return {
            "should_warn": False,
            "reason": "open candle is not moving against reference signal",
            "reference_signal": reference_signal,
            "candle_direction": candle_direction,
            "counter_direction": expected_counter_direction,
            "candle_open": candle_open,
            "current_price": current_price,
            "body_size": body_size,
        }

    zone_context = find_opening_zone_context(candle_open, zones)

    if zone_context.get("zone_distance") is None:
        return {
            "should_warn": False,
            "reason": zone_context.get("reason", "zone context unavailable"),
            "reference_signal": reference_signal,
            "candle_direction": candle_direction,
            "counter_direction": expected_counter_direction,
            "candle_open": candle_open,
            "current_price": current_price,
            "body_size": body_size,
            "upper_zone": zone_context.get("upper_zone"),
            "lower_zone": zone_context.get("lower_zone"),
            "zone_distance": zone_context.get("zone_distance"),
            "zone_context_reason": zone_context.get("reason"),
        }

    zone_distance = float(zone_context["zone_distance"])
    ratio_pct = (body_size / zone_distance) * 100.0

    base_payload = {
        "reference_signal": reference_signal,
        "candle_direction": candle_direction,
        "counter_direction": expected_counter_direction,
        "candle_open": candle_open,
        "current_price": current_price,
        "body_size": body_size,
        "ratio_pct": ratio_pct,
        "early_threshold_pct": float(early_threshold_pct),
        "full_threshold_pct": float(full_threshold_pct),
        "upper_zone": zone_context.get("upper_zone"),
        "lower_zone": zone_context.get("lower_zone"),
        "zone_distance": zone_distance,
        "zone_context_reason": zone_context.get("reason"),
    }

    if ratio_pct < float(early_threshold_pct):
        return {
            "should_warn": False,
            "reason": "below early threshold",
            **base_payload,
        }

    status = "THRESHOLD_REACHED" if ratio_pct >= float(full_threshold_pct) else "EARLY_WARNING"
    opposite_conditions = evaluate_opposite_sma_rsi_conditions(
        df_1h_with_current_open=df_1h_with_current_open,
        reference_signal=reference_signal,
    )

    return {
        "should_warn": True,
        "pair": pair,
        "timeframe": "1h",
        "reference_signal": reference_signal,
        "reference_signal_time": reference_signal_time,
        "candle_open_time": open_1h_row["open_time"].isoformat().replace("+00:00", "Z"),
        "candle_close_time": open_1h_row["close_time"].isoformat().replace("+00:00", "Z"),
        "status": status,
        "opposite_conditions": opposite_conditions,
        **base_payload,
    }


def build_counter_momentum_message(*, event: Dict, streamlit_app_url: str) -> str:
    pair = event.get("pair")

    status = str(event.get("status") or "").upper()
    status_label = "FULL" if status == "THRESHOLD_REACHED" else "EARLY"

    reference_signal = normalize_signal(event.get("reference_signal"))
    counter_direction = normalize_signal(event.get("counter_direction"))

    direction_text = (
        f"{reference_signal[0]}→{counter_direction[0]}"
        if reference_signal in {"LONG", "SHORT"} and counter_direction in {"LONG", "SHORT"}
        else f"{reference_signal}→{counter_direction}"
    )

    ratio_pct = event.get("ratio_pct")
    full_threshold_pct = event.get("full_threshold_pct")
    ratio_text = "NA" if ratio_pct is None else f"{float(ratio_pct):.2f}%"
    full_threshold_text = "NA" if full_threshold_pct is None else f"{float(full_threshold_pct):.2f}%"

    opposite = event.get("opposite_conditions") or {}
    opposite_met = bool(opposite.get("conditions_met"))
    opposite_text = "conditions met ❗" if opposite_met else "not confirmed"

    return "\n".join(
        [
            f"⚠️ {pair} 1H COUNTER {status_label} {direction_text}",
            f"Ref: {reference_signal}",
            f"Counter: {counter_direction}",
            f"Ratio: {ratio_text} / {full_threshold_text}",
            f"Price: {format_price(event.get('current_price'))}",
            f"Opposite signal: {opposite_text}",
            "Candle still open",
        ]
    )

def get_ichimoku_condition_summary(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "long_ok_count": 0,
            "short_ok_count": 0,
            "long_missing": ["not enough data"],
            "short_missing": ["not enough data"],
            "long_check": "0/4 OK — missing: not enough data",
            "short_check": "0/4 OK — missing: not enough data",
            "cloud_top": None,
            "cloud_bottom": None,
            "cloud_range_text": "NA",
            "current_cloud_color": "unknown",
            "future_cloud_color": "unknown",
            "cloud_position": "unknown",
        }

    row = df.iloc[-1]

    has_lag_ref = len(df) >= 27
    if has_lag_ref:
        ref = df.iloc[-27]
        lagging_above = row["close"] > ref["high"]
        lagging_below = row["close"] < ref["low"]
    else:
        lagging_above = False
        lagging_below = False

    tenkan_gt_kijun = row["tenkan"] > row["kijun"]
    tenkan_lt_kijun = row["tenkan"] < row["kijun"]

    a = row.get("senkou_a")
    b = row.get("senkou_b")

    if pd.isna(a) or pd.isna(b):
        cloud_top = None
        cloud_bottom = None
        above_cloud = False
        below_cloud = False
        cloud_position = "unknown"
        current_cloud_color = "unknown"
    else:
        cloud_top = max(float(a), float(b))
        cloud_bottom = min(float(a), float(b))
        close_ = float(row["close"])

        above_cloud = close_ > cloud_top
        below_cloud = close_ < cloud_bottom

        if above_cloud:
            cloud_position = "above cloud"
        elif below_cloud:
            cloud_position = "below cloud"
        else:
            cloud_position = "inside cloud"

        if float(a) > float(b):
            current_cloud_color = "green"
        elif float(a) < float(b):
            current_cloud_color = "red"
        else:
            current_cloud_color = "flat"

    future_a = row.get("senkou_a_base")
    future_b = row.get("senkou_b_base")

    if pd.isna(future_a) or pd.isna(future_b):
        future_green = False
        future_red = False
        future_cloud = "unknown"
    else:
        future_green = float(future_a) > float(future_b)
        future_red = float(future_a) < float(future_b)
        future_cloud = "green" if future_green else "red" if future_red else "flat"

    long_conditions = {
        "Lagging above 26D high": bool(lagging_above),
        "Tenkan>Kijun": bool(tenkan_gt_kijun),
        "Above cloud": bool(above_cloud),
        "Future green": bool(future_green),
    }

    short_conditions = {
        "Lagging below 26D low": bool(lagging_below),
        "Tenkan<Kijun": bool(tenkan_lt_kijun),
        "Below cloud": bool(below_cloud),
        "Future red": bool(future_red),
    }

    long_ok_count = sum(long_conditions.values())
    short_ok_count = sum(short_conditions.values())

    long_missing = [name for name, ok in long_conditions.items() if not ok]
    short_missing = [name for name, ok in short_conditions.items() if not ok]

    long_check = f"{long_ok_count}/4 OK"
    if long_missing:
        long_check += " — missing: " + ", ".join(long_missing)

    short_check = f"{short_ok_count}/4 OK"
    if short_missing:
        short_check += " — missing: " + ", ".join(short_missing)

    cloud_range_text = (
        f"{format_price(cloud_bottom)} - {format_price(cloud_top)}"
        if cloud_top is not None and cloud_bottom is not None
        else "NA"
    )

    return {
        "long_ok_count": long_ok_count,
        "short_ok_count": short_ok_count,
        "long_missing": long_missing,
        "short_missing": short_missing,
        "long_check": long_check,
        "short_check": short_check,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "cloud_range_text": cloud_range_text,
        "current_cloud_color": current_cloud_color,
        "future_cloud_color": future_cloud,
        "cloud_position": cloud_position,
    }


def calculate_ichimoku_trade_plan(
    df: pd.DataFrame,
    signal_type: str,
    tp_multiplier: float,
) -> Dict:
    signal_type = normalize_signal(signal_type)

    if df.empty or signal_type not in {"LONG", "SHORT"}:
        return {
            "valid": False,
            "entry_ref": None,
            "sl_level": None,
            "sl_distance": None,
            "tp_multiplier": float(tp_multiplier),
            "tp_level": None,
            "cloud_top": None,
            "cloud_bottom": None,
            "reason": "no LONG/SHORT signal",
        }

    row = df.iloc[-1]
    summary = get_ichimoku_condition_summary(df)

    close_ = float(row["close"])
    cloud_top = summary.get("cloud_top")
    cloud_bottom = summary.get("cloud_bottom")

    if cloud_top is None or cloud_bottom is None:
        return {
            "valid": False,
            "entry_ref": close_,
            "sl_level": None,
            "sl_distance": None,
            "tp_multiplier": float(tp_multiplier),
            "tp_level": None,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "reason": "cloud boundary missing",
        }

    if signal_type == "LONG":
        sl_level = float(cloud_bottom)
        sl_distance = close_ - sl_level
        tp_level = close_ + float(tp_multiplier) * sl_distance
    else:
        sl_level = float(cloud_top)
        sl_distance = sl_level - close_
        tp_level = close_ - float(tp_multiplier) * sl_distance

    if sl_distance <= 0:
        return {
            "valid": False,
            "entry_ref": close_,
            "sl_level": sl_level,
            "sl_distance": sl_distance,
            "tp_multiplier": float(tp_multiplier),
            "tp_level": None,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "reason": "invalid SL distance",
        }

    return {
        "valid": True,
        "entry_ref": close_,
        "sl_level": sl_level,
        "sl_distance": sl_distance,
        "tp_multiplier": float(tp_multiplier),
        "tp_level": tp_level,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "reason": "ok",
    }


def build_ichimoku_signal_summary_lines(
    df: pd.DataFrame,
    signal_type: str,
    tp_multiplier: float,
    include_trade_plan: bool = True,
) -> List[str]:
    signal_type = normalize_signal(signal_type)
    summary = get_ichimoku_condition_summary(df)

    lines = [
        f"Close: {format_price(df.iloc[-1]['close']) if not df.empty else 'NA'}",
        f"Cloud: {summary['cloud_range_text']}",
        f"Cloud position: {summary['cloud_position']}",
        f"LONG check: {summary['long_check']}",
        f"SHORT check: {summary['short_check']}",
    ]

    if not include_trade_plan or signal_type == "NEUTRAL":
        lines.append("TP/SL: none — no active 1D Ichimoku direction signal")
        return lines

    plan = calculate_ichimoku_trade_plan(df, signal_type, tp_multiplier)

    if not plan["valid"]:
        lines.append(f"TP/SL: unavailable — {plan['reason']}")
        return lines

    lines.extend([
        f"Entry ref: {format_price(plan['entry_ref'])}",
        f"SL: {format_price(plan['sl_level'])}",
        f"Risk: {format_price(plan['sl_distance'])}",
        f"TP {plan['tp_multiplier']:g}R: {format_price(plan['tp_level'])}",
    ])

    return lines


def build_ichimoku_new_signal_message(
    pair: str,
    previous_signal: str | None,
    current_signal: str,
    df_1d_ichi: pd.DataFrame,
    candle_time_1d: str,
    streamlit_app_url: str,
    tp_multiplier: float,
) -> str:
    current_signal = normalize_signal(current_signal)
    previous_signal = normalize_signal(previous_signal)

    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_display(current_signal, previous_signal, True)}",
        "Event: New Ichimoku signal",
        "",
    ])

    lines.extend(
        build_ichimoku_signal_summary_lines(
            df=df_1d_ichi,
            signal_type=current_signal,
            tp_multiplier=tp_multiplier,
            include_trade_plan=True,
        )
    )

    lines.extend([
        f"TP status: waiting confirmation",
        f"Confirmation rule: next 1D close must stay {signal_badge(current_signal)}",
    ])

    return "\n".join(lines)


def build_ichimoku_tp_confirmed_message(
    pair: str,
    trade_state: Dict,
    candle_time_1d: str,
    streamlit_app_url: str,
) -> str:
    signal_type = normalize_signal(trade_state.get("signal_type"))
    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_badge(signal_type)} → {signal_badge(signal_type)} ✅",
        "Event: TP confirmed",
        "",
        f"Original signal: {signal_badge(signal_type)}",
        f"Signal candle: {_format_candle_time_for_message(trade_state.get('signal_time'))}",
        f"Entry ref: {format_price(trade_state.get('entry_ref'))}",
        f"SL: {format_price(trade_state.get('sl_level'))}",
        f"Risk: {format_price(trade_state.get('sl_distance'))}",
        f"TP {format_price(trade_state.get('tp_multiplier'), digits=2)}R: {format_price(trade_state.get('tp_level'))}",
        "TP status: active",
    ])

    return "\n".join(lines)


def build_ichimoku_confirmation_failed_message(
    pair: str,
    trade_state: Dict,
    current_signal: str,
    df_1d_ichi: pd.DataFrame,
    candle_time_1d: str,
    streamlit_app_url: str,
    tp_multiplier: float,
) -> str:
    previous_signal = normalize_signal(trade_state.get("signal_type"))
    current_signal = normalize_signal(current_signal)
    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_badge(previous_signal)} → {signal_badge(current_signal)} ⚠️",
        "Event: TP confirmation failed",
        "",
        f"Previous pending signal: {signal_badge(previous_signal)}",
        "TP status: not activated",
        f"Reason: next 1D close did not stay {signal_badge(previous_signal)}",
        "",
    ])

    lines.extend(
        build_ichimoku_signal_summary_lines(
            df=df_1d_ichi,
            signal_type=current_signal,
            tp_multiplier=tp_multiplier,
            include_trade_plan=False,
        )
    )

    return "\n".join(lines)


def build_ichimoku_tp_hit_message(
    pair: str,
    trade_state: Dict,
    candle_time_1d: str,
    streamlit_app_url: str,
) -> str:
    signal_type = normalize_signal(trade_state.get("signal_type"))
    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_badge(signal_type)} 🎯",
        "Event: TP hit",
        "",
        f"Signal: {signal_badge(signal_type)}",
        f"Signal candle: {_format_candle_time_for_message(trade_state.get('signal_time'))}",
        f"Entry ref: {format_price(trade_state.get('entry_ref'))}",
        f"TP {format_price(trade_state.get('tp_multiplier'), digits=2)}R: {format_price(trade_state.get('tp_level'))}",
        f"Hit candle: {_format_candle_time_for_message(trade_state.get('tp_hit_time') or candle_time_1d)}",
        "Status: target reached",
    ])

    return "\n".join(lines)


def build_ichimoku_state_closed_message(
    pair: str,
    trade_state: Dict,
    current_signal: str,
    candle_time_1d: str,
    streamlit_app_url: str,
    event_title: str,
    note: str,
    suffix: str = "⚠️",
) -> str:
    previous_signal = normalize_signal(trade_state.get("signal_type"))
    current_signal = normalize_signal(current_signal)
    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_badge(previous_signal)} → {signal_badge(current_signal)} {suffix}",
        f"Event: {event_title}",
        "",
        f"Previous active signal: {signal_badge(previous_signal)}",
        f"TP level: {format_price(trade_state.get('tp_level'))}",
        f"TP hit candle: {_format_candle_time_for_message(trade_state.get('tp_hit_time')) if trade_state.get('tp_hit_time') else 'NA'}",
        f"Note: {note}",
    ])

    return "\n".join(lines)


def build_ichimoku_reversal_message(
    pair: str,
    previous_trade_state: Dict | None,
    previous_signal: str | None,
    current_signal: str,
    df_1d_ichi: pd.DataFrame,
    candle_time_1d: str,
    streamlit_app_url: str,
    tp_multiplier: float,
) -> str:
    previous_signal = normalize_signal(previous_signal)
    current_signal = normalize_signal(current_signal)
    lines = [
        f"{pair}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"1d Ichimoku: {signal_badge(previous_signal)} → {signal_badge(current_signal)} 🔁",
        "Event: Ichimoku direction changed",
        "",
    ])

    if previous_trade_state:
        previous_tp_hit = bool(previous_trade_state.get("tp_hit_time"))
        previous_tp_status = "already hit" if previous_tp_hit else "not hit / not confirmed"
        lines.extend([
            f"Previous {previous_signal}:",
            f"Previous TP status: {previous_tp_status}",
            f"Previous TP: {format_price(previous_trade_state.get('tp_level'))}",
            "",
        ])

    lines.extend([
        f"New {current_signal}:",
    ])

    lines.extend(
        build_ichimoku_signal_summary_lines(
            df=df_1d_ichi,
            signal_type=current_signal,
            tp_multiplier=tp_multiplier,
            include_trade_plan=True,
        )
    )

    lines.extend([
        "TP status: waiting confirmation",
        f"Confirmation rule: next 1D close must stay {signal_badge(current_signal)}",
    ])

    return "\n".join(lines)

def build_telegram_message(
    pair: str,
    row_15m: pd.Series,
    signal_15m: str,
    row_1h: pd.Series,
    signal_1h: str,
    prev_signal_1h: str | None,
    row_1d: pd.Series,
    signal_1d: str,
    prev_signal_1d: str | None,
    ichi_details: dict,
    candle_time_15m: str,
    candle_time_1h: str,
    candle_time_1d: str,
    streamlit_app_url: str,
    triggered_15m: bool = False,
    triggered_1h: bool = False,
    triggered_1d: bool = False,
) -> str:
    def fmt(x):
        return format_price(x)
        
    lines = [
        f"{pair}",
        f"15m candle: {_format_candle_time_for_message(candle_time_15m)}",
        f"1h candle: {_format_candle_time_for_message(candle_time_1h)}",
        f"1d candle: {_format_candle_time_for_message(candle_time_1d)}",
    ]

    lines.extend([
        "",
        f"15m: {signal_badge(signal_15m)}",
        f"Close: {fmt(row_15m['close'])}",
        "SMA4 > SMA16" if row_15m["sma4"] > row_15m["sma16"] else "SMA4 < SMA16" if row_15m["sma4"] < row_15m["sma16"] else "SMA4 = SMA16",
        "RSI14 > RSI52" if row_15m["rsi14"] > row_15m["rsi52"] else "RSI14 < RSI52" if row_15m["rsi14"] < row_15m["rsi52"] else "RSI14 = RSI52",
        "",
        f"1h: {signal_display(signal_1h, prev_signal_1h, triggered_1h)}",
        f"Close: {fmt(row_1h['close'])}",
        "SMA4 > SMA16" if row_1h["sma4"] > row_1h["sma16"] else "SMA4 < SMA16" if row_1h["sma4"] < row_1h["sma16"] else "SMA4 = SMA16",
        "RSI14 > RSI52" if row_1h["rsi14"] > row_1h["rsi52"] else "RSI14 < RSI52" if row_1h["rsi14"] < row_1h["rsi52"] else "RSI14 = RSI52",
        "",
        f"1d: {signal_display(signal_1d, prev_signal_1d, triggered_1d)}",
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


def get_closed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only candles whose close_time is <= current UTC time.

    Official snapshot/signal calculations use this closed-candle data.
    Open-candle checks, such as counter momentum early warning, must use
    the latest raw candle separately.
    """
    if df.empty or "close_time" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)

    now_utc = pd.Timestamp.now(tz="UTC")

    work = work[
        work["close_time"].notna()
        & (work["close_time"] <= now_utc)
    ].copy()

    if "open_time" in work.columns:
        work = work.sort_values("open_time").reset_index(drop=True)

    return work


def latest_strategy_snapshot(pair: str, timeframe: str, df: pd.DataFrame) -> Dict:
    closed_df = get_closed_candles(df)

    if closed_df.empty:
        raise ValueError(f"No closed candles available for {pair} {timeframe} snapshot.")

    df_feat = add_ema_rsi_features(closed_df)
    last = df_feat.iloc[-1]

    signal = classify_ema_rsi_signal(last)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "last_open_time": last["open_time"].isoformat().replace("+00:00", "Z"),
        "close": _safe_float(last["close"]),
        "sma4": _safe_float(last.get("sma4")),
        "sma16": _safe_float(last.get("sma16")),
        "sma65": _safe_float(last.get("sma65")),
        "sma120": _safe_float(last.get("sma120")),
        "ema4": _safe_float(last["ema4"]),
        "ema16": _safe_float(last["ema16"]),
        "ema65": _safe_float(last["ema65"]),
        "ema120": _safe_float(last["ema120"]),
        "ema168": _safe_float(last["ema168"]),
        "ema168_adjust_false": _safe_float(last.get("ema168_adjust_false")),
        "ema168_adjust_true": _safe_float(last.get("ema168_adjust_true")),
        "sma168": _safe_float(last.get("sma168")),
        "ema168_sma_seed": _safe_float(last.get("ema168_sma_seed")),
        "ema_calc_candle_count": int(len(df_feat)),
        "ema_calc_first_open_time": (
            df_feat.iloc[0]["open_time"].isoformat().replace("+00:00", "Z")
            if len(df_feat) > 0 and "open_time" in df_feat.columns
            else None
        ),
        "rsi14": _safe_float(last["rsi14"]),
        "rsi52": _safe_float(last["rsi52"]),
        "ema4_slope": _safe_float(last["ema4_slope"]),
        "ema16_slope": _safe_float(last["ema16_slope"]),
        "signal": signal,
        "signal_reason": _signal_reason(last),
        "updated_at": _now_iso(),
    }


def latest_daily_ichimoku_snapshot(pair: str, df: pd.DataFrame) -> Dict:
    # 1D snapshot keeps Ichimoku as the signal source, but also publishes SMA values
    # so dashboard snapshot remains a read-only view of Supabase data.
    daily = add_ema_rsi_features(df)
    daily = add_ichimoku(daily)
    last = daily.iloc[-1]

    signal, ichi_details = classify_ichimoku_signal(daily)

    return {
        "pair": pair,
        "timeframe": "1d",
        "last_open_time": last["open_time"].isoformat().replace("+00:00", "Z"),
        "close": _safe_float(last["close"]),
        "sma4": _safe_float(last.get("sma4")),
        "sma16": _safe_float(last.get("sma16")),
        "sma65": _safe_float(last.get("sma65")),
        "sma120": _safe_float(last.get("sma120")),
        "sma168": _safe_float(last.get("sma168")),
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


def create_supabase_client_from_env() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)


def _is_retryable_supabase_error(exc: Exception) -> bool:
    msg = str(exc).lower()

    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)):
        return True

    retryable_fragments = (
        "connection reset by peer",
        "connection reset",
        "read timed out",
        "write timed out",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "server disconnected",
        "connection aborted",
        "connection refused",
        "remote protocol error",
    )
    return any(fragment in msg for fragment in retryable_fragments)


def _run_with_supabase_retry(action_name: str, func):
    attempt = 0
    delay = SUPABASE_RETRY_BASE_DELAY

    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:
            retryable = _is_retryable_supabase_error(exc)
            is_last_attempt = attempt >= SUPABASE_MAX_RETRIES

            if (not retryable) or is_last_attempt:
                raise

            log(
                f"[Retry] {action_name} failed on attempt {attempt}/{SUPABASE_MAX_RETRIES}: "
                f"{type(exc).__name__}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, SUPABASE_RETRY_MAX_DELAY)


def upsert_in_chunks(supabase: Client, table_name: str, rows: Iterable[Dict], on_conflict: str) -> None:
    rows = list(rows)
    if not rows:
        return

    for i in range(0, len(rows), UPSERT_CHUNK_SIZE):
        chunk = rows[i:i + UPSERT_CHUNK_SIZE]

        def _do_upsert():
            return (
                supabase
                .table(table_name)
                .upsert(chunk, on_conflict=on_conflict, returning="minimal")
                .execute()
            )

        _run_with_supabase_retry(
            action_name=f"upsert {table_name} chunk {i // UPSERT_CHUNK_SIZE + 1}",
            func=_do_upsert,
        )


def get_previous_snapshot_map(supabase: Client, pair: str) -> Dict[str, Dict]:
    def _do_fetch():
        return (
            supabase.table("futures_signal_snapshots")
            .select("pair,timeframe,signal")
            .eq("pair", pair)
            .execute()
        )

    resp = _run_with_supabase_retry(
        action_name=f"fetch previous snapshots for {pair}",
        func=_do_fetch,
    )
    rows = resp.data or []
    return {row["timeframe"]: row for row in rows}


def send_telegram_message(text: str, bot_token: str, chat_ids: List[str]) -> None:
    if not bot_token or not chat_ids:
        log("Notification credentials missing, skipping Telegram send.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": text}

        last_exc = None
        delay = 1.0
        masked_chat = f"...{str(chat_id)[-4:]}" if chat_id else "unknown"

        for attempt in range(1, 5):
            try:
                r = requests.post(url, json=payload, timeout=20)
                r.raise_for_status()
                log(f"Notification sent to {masked_chat}")
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == 4:
                    break
                log(
                    f"[Retry] notification send failed for {masked_chat} "
                    f"on attempt {attempt}/4. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        if last_exc is not None:
            raise RuntimeError(f"Notification send failed after 4 attempts for {masked_chat}")


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


def _build_app_link(pair: str, streamlit_app_url: str) -> str:
    if not streamlit_app_url:
        return ""
    separator = "&" if "?" in streamlit_app_url else "?"
    return f"{streamlit_app_url}{separator}pair={pair}"


def _signal_reason(last: pd.Series) -> str:
    if bool(last.get("long_signal", False)):
        return "SMA4 crossed above SMA16 and RSI14 > RSI52 and RSI14 >= 50"

    if bool(last.get("short_signal", False)):
        return "SMA4 crossed below SMA16 and RSI14 < RSI52 and RSI14 <= 50"

    return "No fresh SMA crossover signal"
