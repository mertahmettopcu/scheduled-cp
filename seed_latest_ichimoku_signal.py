#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from core_utils import (
    add_ichimoku,
    calculate_ichimoku_trade_plan,
    classify_ichimoku_signal,
    create_supabase_client_from_env,
    log,
    normalize_signal,
    read_ichimoku_tp_multiplier,
)


ACTIVE_STATUSES = [
    "PENDING_CONFIRMATION",
    "ACTIVE",
    "TP_HIT",
]


def _iso_from_ts(value) -> str:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_1d_candles_from_supabase(supabase, pair: str) -> pd.DataFrame:
    resp = (
        supabase
        .table("futures_candles")
        .select("*")
        .eq("pair", pair)
        .eq("timeframe", "1d")
        .order("open_time", desc=True)
        .limit(600)
        .execute()
    )

    df = pd.DataFrame(resp.data or [])

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df
        .dropna(subset=["open_time", "close_time", "open", "high", "low", "close"])
        .sort_values("open_time")
        .drop_duplicates(subset=["open_time"])
        .reset_index(drop=True)
    )

    return df


def keep_closed_daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    now_utc = pd.Timestamp.utcnow()

    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")

    out = df[df["close_time"] <= now_utc].copy()
    return out.sort_values("open_time").reset_index(drop=True)


def build_ichimoku_signal_series(df_1d: pd.DataFrame) -> pd.DataFrame:
    ichi = add_ichimoku(df_1d)

    signals: List[str] = []

    for i in range(len(ichi)):
        partial = ichi.iloc[: i + 1].copy()
        signal, _ = classify_ichimoku_signal(partial)
        signals.append(normalize_signal(signal))

    out = ichi.copy()
    out["ichimoku_signal"] = signals
    out["previous_ichimoku_signal"] = out["ichimoku_signal"].shift(1).fillna("NEUTRAL")

    out["new_directional_signal"] = (
        out["ichimoku_signal"].isin(["LONG", "SHORT"])
        & (out["previous_ichimoku_signal"] != out["ichimoku_signal"])
    )

    return out


def fetch_open_states(supabase, pair: str) -> List[Dict]:
    resp = (
        supabase
        .table("ichimoku_trade_states")
        .select("*")
        .eq("pair", pair)
        .in_("status", ACTIVE_STATUSES)
        .order("signal_time", desc=True)
        .execute()
    )

    return resp.data or []


def seed_latest_signal() -> None:
    pair = os.getenv("SEED_PAIR", "BTCUSDT").strip().upper()
    max_signal_age_days = float(os.getenv("SEED_MAX_SIGNAL_AGE_DAYS", "3"))
    tp_multiplier = read_ichimoku_tp_multiplier(default=1.7)

    supabase = create_supabase_client_from_env()

    open_states = fetch_open_states(supabase, pair)

    if open_states:
        log(f"Open Ichimoku trade state already exists for {pair}. Seed skipped.")
        for state in open_states:
            log(
                f"  └─ id={state.get('id')} "
                f"signal={state.get('signal_type')} "
                f"status={state.get('status')} "
                f"signal_time={state.get('signal_time')}"
            )
        return

    raw_1d = load_1d_candles_from_supabase(supabase, pair)

    if raw_1d.empty:
        raise SystemExit(f"No 1D candles found in futures_candles for {pair}.")

    closed_1d = keep_closed_daily_candles(raw_1d)

    if closed_1d.empty:
        raise SystemExit(f"No closed 1D candles available for {pair}.")

    signal_df = build_ichimoku_signal_series(closed_1d)

    candidates = signal_df[signal_df["new_directional_signal"] == True].copy()

    if candidates.empty:
        raise SystemExit(f"No closed 1D Ichimoku LONG/SHORT signal found for {pair}.")

    latest_signal_row = candidates.iloc[-1]
    latest_signal_idx = latest_signal_row.name

    signal_type = normalize_signal(latest_signal_row["ichimoku_signal"])
    signal_time = pd.Timestamp(latest_signal_row["open_time"])

    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")

    signal_age_days = (now_utc - signal_time).total_seconds() / 86400

    if signal_age_days > max_signal_age_days:
        raise SystemExit(
            f"Latest Ichimoku signal is older than allowed seed window. "
            f"signal_time={signal_time.isoformat()}, "
            f"age_days={signal_age_days:.2f}, "
            f"max_signal_age_days={max_signal_age_days:.2f}. "
            f"Seed aborted."
        )

    plan_df = signal_df.iloc[: latest_signal_idx + 1].copy().reset_index(drop=True)

    plan = calculate_ichimoku_trade_plan(
        df=plan_df,
        signal_type=signal_type,
        tp_multiplier=tp_multiplier,
    )

    if not plan.get("valid"):
        raise SystemExit(
            f"Ichimoku TP plan invalid for {pair} {signal_type}: {plan.get('reason')}"
        )

    row = {
        "pair": pair,
        "signal_type": signal_type,
        "signal_time": _iso_from_ts(latest_signal_row["open_time"]),
        "signal_close": _safe_float(latest_signal_row["close"]),
        "entry_ref": _safe_float(plan["entry_ref"]),
        "cloud_top": _safe_float(plan["cloud_top"]),
        "cloud_bottom": _safe_float(plan["cloud_bottom"]),
        "sl_level": _safe_float(plan["sl_level"]),
        "sl_distance": _safe_float(plan["sl_distance"]),
        "tp_multiplier": _safe_float(plan["tp_multiplier"]),
        "tp_level": _safe_float(plan["tp_level"]),
        "status": "PENDING_CONFIRMATION",
    }

    resp = (
        supabase
        .table("ichimoku_trade_states")
        .upsert(
            row,
            on_conflict="pair,signal_time,signal_type",
        )
        .execute()
    )

    data = resp.data or []

    log(f"Seeded latest Ichimoku signal for {pair}")
    log(f"  └─ signal_type: {signal_type}")
    log(f"  └─ signal_time: {row['signal_time']}")
    log(f"  └─ status: PENDING_CONFIRMATION")
    log(f"  └─ entry_ref: {row['entry_ref']}")
    log(f"  └─ sl_level: {row['sl_level']}")
    log(f"  └─ tp_multiplier: {row['tp_multiplier']}")
    log(f"  └─ tp_level: {row['tp_level']}")

    if data:
        log(f"  └─ inserted/updated id: {data[0].get('id')}")


if __name__ == "__main__":
    seed_latest_signal()
