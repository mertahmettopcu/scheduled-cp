#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from crypto_futures_common import (
    add_ema_rsi_features,
    add_ichimoku,
    build_telegram_message,
    candles_to_records,
    classify_ichimoku_signal,
    create_supabase_client_from_env,
    get_previous_snapshot_map,
    klines_to_df,
    latest_daily_ichimoku_snapshot,
    latest_strategy_snapshot,
    log,
    send_telegram_message,
    signal_changed,
    upsert_in_chunks,
)

INPUT_FILE = Path("binance_data.json")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS_RAW = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [x.strip() for x in TELEGRAM_CHAT_IDS_RAW.split(",") if x.strip()]


def load_cached_raw_data() -> dict:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing {INPUT_FILE}. Fetch stage must run before process stage.")
    return json.loads(INPUT_FILE.read_text(encoding="utf-8"))


def run() -> None:
    supabase = create_supabase_client_from_env()
    raw_payload = load_cached_raw_data()

    # ✅ CRITICAL FIX: sadece fetch edilen coinleri kullan
    pairs_to_process = list(raw_payload.get("pairs", {}).keys())

    if not pairs_to_process:
        raise SystemExit("No pairs found in raw payload. Fetch stage likely failed.")

    candle_rows: List[Dict] = []
    snapshot_rows: List[Dict] = []

    for pair in pairs_to_process:
        log(f"• Processing {pair}")

        pair_raw = raw_payload.get("pairs", {}).get(pair, {})
        pair_dfs: Dict[str, pd.DataFrame] = {}

        for timeframe, raw in pair_raw.items():
            df = klines_to_df(pair, timeframe, raw)
            if df.empty:
                log(f"  └─ {timeframe}: empty")
                continue

            pair_dfs[timeframe] = df
            candle_rows.extend(candles_to_records(df))
            log(f"  └─ {timeframe}: prepared {len(df)} candles")

        if "15m" not in pair_dfs or "1h" not in pair_dfs or "1d" not in pair_dfs:
            log(f"  └─ Skipping {pair} snapshot generation because one or more timeframes are missing")
            continue

        snap_15m = latest_strategy_snapshot(pair, "15m", pair_dfs["15m"])
        snap_1h = latest_strategy_snapshot(pair, "1h", pair_dfs["1h"])
        snap_1d = latest_daily_ichimoku_snapshot(pair, pair_dfs["1d"])

        snapshot_rows.extend([snap_15m, snap_1h, snap_1d])

        prev_map = get_previous_snapshot_map(supabase, pair)
        prev_15m = prev_map.get("15m")
        prev_1h = prev_map.get("1h")
        prev_1d = prev_map.get("1d")

        changed_15m = signal_changed(prev_15m.get("signal") if prev_15m else None, snap_15m["signal"])
        changed_1h = signal_changed(prev_1h.get("signal") if prev_1h else None, snap_1h["signal"])
        changed_1d = signal_changed(prev_1d.get("signal") if prev_1d else None, snap_1d["signal"])

        log(
            f"  └─ Signal compare {pair} | "
            f"15m: prev={prev_15m.get('signal') if prev_15m else None} new={snap_15m['signal']} changed={changed_15m} | "
            f"1h: prev={prev_1h.get('signal') if prev_1h else None} new={snap_1h['signal']} changed={changed_1h} | "
            f"1d: prev={prev_1d.get('signal') if prev_1d else None} new={snap_1d['signal']} changed={changed_1d}"
        )

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

            _, ichi_details = classify_ichimoku_signal(df_1d_ichi)

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
                streamlit_app_url=STREAMLIT_APP_URL,
                triggered_15m=changed_15m,
                triggered_1h=changed_1h,
                triggered_1d=changed_1d,
            )
            send_telegram_message(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS)
            log(f"  └─ 📨 Telegram sent for {pair}")
        else:
            log(f"  └─ Telegram skipped for {pair} (no new signal)")

    if not candle_rows:
        raise SystemExit("No candle rows prepared; nothing to upsert.")

    dedup_candles = {(r["pair"], r["timeframe"], r["open_time"]): r for r in candle_rows}
    dedup_snapshots = {(r["pair"], r["timeframe"]): r for r in snapshot_rows}

    log(f"⬆️ Upserting {len(dedup_candles)} candle rows into futures_candles")
    upsert_in_chunks(supabase, "futures_candles", dedup_candles.values(), on_conflict="pair,timeframe,open_time")

    log(f"⬆️ Upserting {len(dedup_snapshots)} snapshot rows into futures_signal_snapshots")
    upsert_in_chunks(supabase, "futures_signal_snapshots", dedup_snapshots.values(), on_conflict="pair,timeframe")

    log("✅ Process/upload stage complete")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
