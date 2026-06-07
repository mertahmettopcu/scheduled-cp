#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core_utils import (
    add_ema_rsi_features,
    add_ichimoku,
    build_ichimoku_confirmation_failed_message,
    build_ichimoku_new_signal_message,
    build_ichimoku_reversal_message,
    build_ichimoku_state_closed_message,
    build_ichimoku_tp_confirmed_message,
    build_ichimoku_tp_hit_message,
    build_telegram_message,
    calculate_ichimoku_trade_plan,
    candles_to_records,
    classify_ichimoku_signal,
    create_supabase_client_from_env,
    get_closed_candles,
    get_previous_snapshot_map,
    klines_to_df,
    latest_daily_ichimoku_snapshot,
    latest_strategy_snapshot,
    log,
    normalize_signal,
    read_ichimoku_tp_multiplier,
    send_telegram_message,
    signal_changed,
    upsert_in_chunks,
)

INPUT_FILE = Path("binance_data.json")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS_RAW = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [x.strip() for x in TELEGRAM_CHAT_IDS_RAW.split(",") if x.strip()]
ICHIMOKU_TP_MULTIPLIER = read_ichimoku_tp_multiplier(default=1.7)


def load_cached_raw_data() -> dict:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing {INPUT_FILE}. Fetch stage must run before process stage.")
    return json.loads(INPUT_FILE.read_text(encoding="utf-8"))


def fetch_open_ichimoku_trade_states(supabase, pair: str) -> List[Dict]:
    """
    Fetch non-final Ichimoku 1D trade lifecycle states for this pair.

    Final states are intentionally excluded:
    - CONFIRMATION_FAILED
    - INVALIDATED_BEFORE_TP
    - ENDED_AFTER_TP
    - REVERSED_BEFORE_TP
    - REVERSED_AFTER_TP

    TP_HIT is kept open until the Ichimoku state later ends or reverses,
    because we still want to send a different close message after TP was hit.
    """
    active_statuses = [
        "PENDING_CONFIRMATION",
        "ACTIVE",
        "TP_HIT",
    ]

    resp = (
        supabase
        .table("ichimoku_trade_states")
        .select("*")
        .eq("pair", pair)
        .in_("status", active_statuses)
        .order("signal_time", desc=True)
        .execute()
    )

    return resp.data or []


def get_latest_open_ichimoku_trade_state(supabase, pair: str) -> Dict | None:
    rows = fetch_open_ichimoku_trade_states(supabase, pair)
    return rows[0] if rows else None


def insert_ichimoku_trade_state(
    supabase,
    *,
    pair: str,
    signal_type: str,
    signal_time: str,
    signal_close: float,
    entry_ref: float,
    cloud_top: float,
    cloud_bottom: float,
    sl_level: float,
    sl_distance: float,
    tp_multiplier: float,
    tp_level: float,
) -> Dict | None:
    row = {
        "pair": pair,
        "signal_type": signal_type,
        "signal_time": signal_time,
        "signal_close": signal_close,
        "entry_ref": entry_ref,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "sl_level": sl_level,
        "sl_distance": sl_distance,
        "tp_multiplier": tp_multiplier,
        "tp_level": tp_level,
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
    return data[0] if data else None


def update_ichimoku_trade_state(
    supabase,
    state_id: int,
    updates: Dict,
) -> Dict | None:
    payload = dict(updates)
    payload["updated_at"] = pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z")

    resp = (
        supabase
        .table("ichimoku_trade_states")
        .update(payload)
        .eq("id", state_id)
        .execute()
    )

    data = resp.data or []
    return data[0] if data else None


def _iso_from_ts(value) -> str:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def get_closed_daily_ichimoku_df(df_1d_ichi: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only fully closed 1D candles.

    Binance can return the currently open daily candle as the last row.
    Ichimoku lifecycle confirmation must use the latest CLOSED daily candle,
    not the currently forming candle.
    """
    if df_1d_ichi.empty or "close_time" not in df_1d_ichi.columns:
        return pd.DataFrame()

    work = df_1d_ichi.copy()
    work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)

    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")

    work = work[
        work["close_time"].notna()
        & (work["close_time"] <= now_utc)
    ].copy()

    return work.sort_values("open_time").reset_index(drop=True)


def _state_is_opposite(state_signal: str | None, current_signal: str | None) -> bool:
    state_signal = normalize_signal(state_signal)
    current_signal = normalize_signal(current_signal)

    return (
        state_signal in {"LONG", "SHORT"}
        and current_signal in {"LONG", "SHORT"}
        and state_signal != current_signal
    )


def _tp_hit_for_state(row_1d: pd.Series, state: Dict) -> bool:
    signal_type = normalize_signal(state.get("signal_type"))
    tp_level = state.get("tp_level")

    if tp_level is None or pd.isna(tp_level):
        return False

    tp_level = float(tp_level)

    if signal_type == "LONG":
        return bool(float(row_1d["high"]) >= tp_level)

    if signal_type == "SHORT":
        return bool(float(row_1d["low"]) <= tp_level)

    return False


def _build_trade_state_insert_payload(
    pair: str,
    signal_type: str,
    latest_1d: pd.Series,
    df_1d_ichi: pd.DataFrame,
    tp_multiplier: float,
) -> Dict | None:
    plan = calculate_ichimoku_trade_plan(
        df=df_1d_ichi,
        signal_type=signal_type,
        tp_multiplier=tp_multiplier,
    )

    if not plan.get("valid"):
        log(
            f"  └─ Ichimoku trade plan skipped for {pair}: "
            f"{signal_type} plan invalid ({plan.get('reason')})"
        )
        return None

    return {
        "pair": pair,
        "signal_type": signal_type,
        "signal_time": _iso_from_ts(latest_1d["open_time"]),
        "signal_close": float(latest_1d["close"]),
        "entry_ref": float(plan["entry_ref"]),
        "cloud_top": float(plan["cloud_top"]),
        "cloud_bottom": float(plan["cloud_bottom"]),
        "sl_level": float(plan["sl_level"]),
        "sl_distance": float(plan["sl_distance"]),
        "tp_multiplier": float(plan["tp_multiplier"]),
        "tp_level": float(plan["tp_level"]),
    }


def handle_pending_ichimoku_state(
    supabase,
    *,
    pair: str,
    state: Dict,
    current_signal_1d: str,
    latest_1d: pd.Series,
    df_1d_ichi: pd.DataFrame,
    candle_time_1d: str,
) -> List[str]:
    """
    Handles PENDING_CONFIRMATION state.

    If next 1D close stays in same direction:
    PENDING_CONFIRMATION -> ACTIVE

    If not:
    PENDING_CONFIRMATION -> CONFIRMATION_FAILED

    Returns Telegram message texts.
    """
    messages: List[str] = []

    state_id = state["id"]
    state_signal = normalize_signal(state.get("signal_type"))
    current_signal = normalize_signal(current_signal_1d)

    # Aynı sinyal mumunda confirmation kontrolü yapılmasın.
    # Confirmation için en az bir sonraki daily mum gerekir.
    state_signal_time = pd.Timestamp(state["signal_time"])
    current_open_time = pd.Timestamp(latest_1d["open_time"])

    if state_signal_time.tzinfo is None:
        state_signal_time = state_signal_time.tz_localize("UTC")

    if current_open_time.tzinfo is None:
        current_open_time = current_open_time.tz_localize("UTC")

    if current_open_time <= state_signal_time:
        return messages

    # Kritik güvenlik:
    # Confirmation / failed confirmation sadece gerçekten kapanmış daily mumla üretilecek.
    current_close_time = pd.Timestamp(latest_1d["close_time"])

    if current_close_time.tzinfo is None:
        current_close_time = current_close_time.tz_localize("UTC")

    now_utc = pd.Timestamp.utcnow()

    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")

    if now_utc < current_close_time:
        log(
            f"  └─ Ichimoku pending confirmation waiting for daily close "
            f"for {pair}: close_time={current_close_time.isoformat()}"
        )
        return messages

    if current_signal == state_signal:
        updated = update_ichimoku_trade_state(
            supabase,
            state_id,
            {
                "status": "ACTIVE",
                "confirmation_candle_time": _iso_from_ts(latest_1d["open_time"]),
                "confirmation_notified": True,
            },
        ) or state

        messages.append(
            build_ichimoku_tp_confirmed_message(
                pair=pair,
                trade_state=updated,
                candle_time_1d=candle_time_1d,
                streamlit_app_url=STREAMLIT_APP_URL,
            )
        )

        log(f"  └─ Ichimoku TP confirmed for {pair}")
        return messages

    updated = update_ichimoku_trade_state(
        supabase,
        state_id,
        {
            "status": "CONFIRMATION_FAILED",
            "closed_at": _iso_from_ts(latest_1d["open_time"]),
            "close_reason": "NEXT_DAILY_CLOSE_DID_NOT_CONFIRM",
            "confirmation_failed_notified": True,
        },
    ) or state

    messages.append(
        build_ichimoku_confirmation_failed_message(
            pair=pair,
            trade_state=updated,
            current_signal=current_signal,
            df_1d_ichi=df_1d_ichi,
            candle_time_1d=candle_time_1d,
            streamlit_app_url=STREAMLIT_APP_URL,
            tp_multiplier=ICHIMOKU_TP_MULTIPLIER,
        )
    )

    log(f"  └─ Ichimoku TP confirmation failed for {pair}")
    return messages


def handle_active_ichimoku_state(
    supabase,
    *,
    pair: str,
    state: Dict,
    current_signal_1d: str,
    latest_1d: pd.Series,
    candle_time_1d: str,
) -> List[str]:
    """
    Handles ACTIVE or TP_HIT state.

    ACTIVE:
    - If TP hit: ACTIVE -> TP_HIT
    - If state becomes NEUTRAL before TP: ACTIVE -> INVALIDATED_BEFORE_TP
    - If opposite signal before TP: ACTIVE -> REVERSED_BEFORE_TP

    TP_HIT:
    - If state becomes NEUTRAL after TP: TP_HIT -> ENDED_AFTER_TP
    - If opposite signal after TP: TP_HIT -> REVERSED_AFTER_TP
    """
    messages: List[str] = []

    state_id = state["id"]
    status = str(state.get("status") or "").upper()
    state_signal = normalize_signal(state.get("signal_type"))
    current_signal = normalize_signal(current_signal_1d)

    tp_already_hit = status == "TP_HIT" or bool(state.get("tp_hit_time"))

    if status == "ACTIVE" and _tp_hit_for_state(latest_1d, state):
        updated = update_ichimoku_trade_state(
            supabase,
            state_id,
            {
                "status": "TP_HIT",
                "tp_hit_time": _iso_from_ts(latest_1d["open_time"]),
                "tp_hit_notified": True,
            },
        ) or state

        messages.append(
            build_ichimoku_tp_hit_message(
                pair=pair,
                trade_state=updated,
                candle_time_1d=candle_time_1d,
                streamlit_app_url=STREAMLIT_APP_URL,
            )
        )

        log(f"  └─ Ichimoku TP hit for {pair}")
        return messages

    if current_signal == state_signal:
        return messages

    if current_signal == "NEUTRAL":
        if tp_already_hit:
            updated = update_ichimoku_trade_state(
                supabase,
                state_id,
                {
                    "status": "ENDED_AFTER_TP",
                    "closed_at": _iso_from_ts(latest_1d["open_time"]),
                    "close_reason": "STATE_ENDED_AFTER_TP",
                    "close_notified": True,
                },
            ) or state

            messages.append(
                build_ichimoku_state_closed_message(
                    pair=pair,
                    trade_state=updated,
                    current_signal=current_signal,
                    candle_time_1d=candle_time_1d,
                    streamlit_app_url=STREAMLIT_APP_URL,
                    event_title="Signal ended after TP hit",
                    note="Ichimoku state ended after target was reached.",
                    suffix="ℹ️",
                )
            )

            log(f"  └─ Ichimoku state ended after TP for {pair}")
            return messages

        updated = update_ichimoku_trade_state(
            supabase,
            state_id,
            {
                "status": "INVALIDATED_BEFORE_TP",
                "closed_at": _iso_from_ts(latest_1d["open_time"]),
                "close_reason": "STATE_INVALIDATED_BEFORE_TP",
                "close_notified": True,
            },
        ) or state

        messages.append(
            build_ichimoku_state_closed_message(
                pair=pair,
                trade_state=updated,
                current_signal=current_signal,
                candle_time_1d=candle_time_1d,
                streamlit_app_url=STREAMLIT_APP_URL,
                event_title="Confirmed signal invalidated before TP",
                note="Confirmed Ichimoku context ended before TP was reached.",
                suffix="⚠️",
            )
        )

        log(f"  └─ Ichimoku active state invalidated before TP for {pair}")
        return messages

    # Opposite LONG/SHORT reversal closure. Yeni sinyal mesajı aşağıda birleşik olarak üretilecek.
    if _state_is_opposite(state_signal, current_signal):
        if tp_already_hit:
            update_ichimoku_trade_state(
                supabase,
                state_id,
                {
                    "status": "REVERSED_AFTER_TP",
                    "closed_at": _iso_from_ts(latest_1d["open_time"]),
                    "close_reason": "OPPOSITE_SIGNAL_AFTER_TP",
                    "close_notified": True,
                },
            )
            log(f"  └─ Ichimoku state reversed after TP for {pair}")
            return messages

        update_ichimoku_trade_state(
            supabase,
            state_id,
            {
                "status": "REVERSED_BEFORE_TP",
                "closed_at": _iso_from_ts(latest_1d["open_time"]),
                "close_reason": "OPPOSITE_SIGNAL_BEFORE_TP",
                "close_notified": True,
            },
        )
        log(f"  └─ Ichimoku state reversed before TP for {pair}")
        return messages

    return messages


def run() -> None:
    supabase = create_supabase_client_from_env()
    raw_payload = load_cached_raw_data()

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

        closed_15m_df = get_closed_candles(pair_dfs["15m"])
        closed_1h_df = get_closed_candles(pair_dfs["1h"])
        closed_1d_df = get_closed_candles(pair_dfs["1d"])

        if closed_15m_df.empty or closed_1h_df.empty or closed_1d_df.empty:
            log(f"  └─ Skipping {pair} snapshot generation because one or more closed timeframes are missing")
            continue

        open_1h_df = pair_dfs["1h"].copy().sort_values("open_time").reset_index(drop=True)
        open_1h_row = open_1h_df.iloc[-1]
        latest_closed_1h_row = closed_1h_df.iloc[-1]

        if pd.Timestamp(open_1h_row["open_time"]) > pd.Timestamp(latest_closed_1h_row["open_time"]):
            log(
                f"  └─ 1h open candle detected for {pair}: "
                f"open_time={_iso_from_ts(open_1h_row['open_time'])}; "
                f"official 1h signal uses closed candle={_iso_from_ts(latest_closed_1h_row['open_time'])}"
            )
        else:
            log(
                f"  └─ No separate open 1h candle detected for {pair}; "
                f"official 1h signal uses latest candle={_iso_from_ts(latest_closed_1h_row['open_time'])}"
            )

        snap_15m = latest_strategy_snapshot(pair, "15m", closed_15m_df)
        snap_1h = latest_strategy_snapshot(pair, "1h", closed_1h_df)
        snap_1d = latest_daily_ichimoku_snapshot(pair, closed_1d_df)

        snapshot_rows.extend([snap_15m, snap_1h, snap_1d])

        prev_map = get_previous_snapshot_map(supabase, pair)
        prev_15m = prev_map.get("15m")
        prev_1h = prev_map.get("1h")
        prev_1d = prev_map.get("1d")

        changed_15m = signal_changed(prev_15m.get("signal") if prev_15m else None, snap_15m["signal"])
        changed_1h = signal_changed(prev_1h.get("signal") if prev_1h else None, snap_1h["signal"])
        changed_1d = signal_changed(prev_1d.get("signal") if prev_1d else None, snap_1d["signal"])

        log(f"  └─ Signal state checked for {pair}")

        df_15m_feat = add_ema_rsi_features(closed_15m_df)
        df_1h_feat = add_ema_rsi_features(closed_1h_df)
        df_1d_ichi = add_ichimoku(closed_1d_df)

        latest_15m = df_15m_feat.iloc[-1]
        latest_1h = df_1h_feat.iloc[-1]
        latest_1d = df_1d_ichi.iloc[-1]

        _, ichi_details = classify_ichimoku_signal(df_1d_ichi)

        df_1d_ichi_lifecycle = get_closed_daily_ichimoku_df(df_1d_ichi)

        if df_1d_ichi_lifecycle.empty:
            log(f"  └─ Ichimoku lifecycle skipped for {pair}: no closed 1D candle")
            latest_1d_lifecycle = latest_1d
            current_signal_1d_lifecycle = normalize_signal(snap_1d["signal"])
            candle_time_1d_lifecycle = snap_1d["last_open_time"]
        else:
            latest_1d_lifecycle = df_1d_ichi_lifecycle.iloc[-1]
            current_signal_1d_lifecycle, _ = classify_ichimoku_signal(df_1d_ichi_lifecycle)
            current_signal_1d_lifecycle = normalize_signal(current_signal_1d_lifecycle)
            candle_time_1d_lifecycle = _iso_from_ts(latest_1d_lifecycle["open_time"])

        prev_signal_1h = prev_1h.get("signal") if prev_1h else None
        prev_signal_1d = prev_1d.get("signal") if prev_1d else None

        current_signal_1d = current_signal_1d_lifecycle
        previous_signal_1d = normalize_signal(prev_signal_1d)

        changed_1d_lifecycle = signal_changed(previous_signal_1d, current_signal_1d)

        notification_messages: List[str] = []

        has_previous_snapshot = (
            prev_15m is not None
            or prev_1h is not None
            or prev_1d is not None
        )

        # -------------------------------------------------
        # 1H legacy notification
        # -------------------------------------------------
        # 1H tarafını şimdilik eski genel mesaj formatıyla koruyoruz.
        # 1D değişimi için özel Ichimoku lifecycle mesajları aşağıda üretilecek.
        triggered_1h_directional = (
            has_previous_snapshot
            and changed_1h
            and normalize_signal(snap_1h["signal"]) in {"LONG", "SHORT"}
        )

        if triggered_1h_directional:
            legacy_1h_message = build_telegram_message(
                pair=pair,
                row_15m=latest_15m,
                signal_15m=snap_15m["signal"],
                row_1h=latest_1h,
                signal_1h=snap_1h["signal"],
                prev_signal_1h=prev_signal_1h,
                row_1d=latest_1d,
                signal_1d=snap_1d["signal"],
                prev_signal_1d=prev_signal_1d,
                ichi_details=ichi_details,
                candle_time_15m=snap_15m["last_open_time"],
                candle_time_1h=snap_1h["last_open_time"],
                candle_time_1d=snap_1d["last_open_time"],
                streamlit_app_url=STREAMLIT_APP_URL,
                triggered_15m=False,
                triggered_1h=triggered_1h_directional,
                triggered_1d=False,
            )
            notification_messages.append(legacy_1h_message)

        # -------------------------------------------------
        # 1D Ichimoku lifecycle
        # -------------------------------------------------
        latest_open_state = get_latest_open_ichimoku_trade_state(supabase, pair)
        previous_trade_for_reversal = latest_open_state

        if latest_open_state:
            state_status = str(latest_open_state.get("status") or "").upper()

            if state_status == "PENDING_CONFIRMATION":
                notification_messages.extend(
                    handle_pending_ichimoku_state(
                        supabase,
                        pair=pair,
                        state=latest_open_state,
                        current_signal_1d=current_signal_1d,
                        latest_1d=latest_1d_lifecycle,
                        df_1d_ichi=df_1d_ichi_lifecycle,
                        candle_time_1d=candle_time_1d_lifecycle,
                    )
                )

                # State güncellenmiş olabilir; yeni sinyal kontrolü için tekrar oku.
                latest_open_state = get_latest_open_ichimoku_trade_state(supabase, pair)
                previous_trade_for_reversal = previous_trade_for_reversal or latest_open_state

            elif state_status in {"ACTIVE", "TP_HIT"}:
                notification_messages.extend(
                    handle_active_ichimoku_state(
                        supabase,
                        pair=pair,
                        state=latest_open_state,
                        current_signal_1d=current_signal_1d,
                        latest_1d=latest_1d_lifecycle,
                        candle_time_1d=candle_time_1d_lifecycle,
                    )
                )

                # State güncellenmiş olabilir; yeni sinyal kontrolü için tekrar oku.
                latest_open_state = get_latest_open_ichimoku_trade_state(supabase, pair)

        # -------------------------------------------------
        # New 1D Ichimoku LONG/SHORT signal
        # -------------------------------------------------
        new_directional_1d_signal = (
            has_previous_snapshot
            and changed_1d_lifecycle
            and current_signal_1d in {"LONG", "SHORT"}
        )

        if new_directional_1d_signal:
            payload = _build_trade_state_insert_payload(
                pair=pair,
                signal_type=current_signal_1d,
                latest_1d=latest_1d_lifecycle,
                df_1d_ichi=df_1d_ichi_lifecycle,
                tp_multiplier=ICHIMOKU_TP_MULTIPLIER,
            )

            if payload is not None:
                inserted_state = insert_ichimoku_trade_state(
                    supabase,
                    **payload,
                )

                # Eğer zıt yöne geçiş varsa birleşik reversal mesajı gönder.
                # Yoksa normal new signal mesajı gönder.
                if previous_signal_1d in {"LONG", "SHORT"} and previous_signal_1d != current_signal_1d:
                    notification_messages.append(
                        build_ichimoku_reversal_message(
                            pair=pair,
                            previous_trade_state=previous_trade_for_reversal,
                            previous_signal=previous_signal_1d,
                            current_signal=current_signal_1d,
                            df_1d_ichi=df_1d_ichi_lifecycle,
                            candle_time_1d=candle_time_1d_lifecycle,
                            streamlit_app_url=STREAMLIT_APP_URL,
                            tp_multiplier=ICHIMOKU_TP_MULTIPLIER,
                        )
                    )
                else:
                    notification_messages.append(
                        build_ichimoku_new_signal_message(
                            pair=pair,
                            previous_signal=previous_signal_1d,
                            current_signal=current_signal_1d,
                            df_1d_ichi=df_1d_ichi_lifecycle,
                            candle_time_1d=candle_time_1d_lifecycle,
                            streamlit_app_url=STREAMLIT_APP_URL,
                            tp_multiplier=ICHIMOKU_TP_MULTIPLIER,
                        )
                    )

                log(f"  └─ Ichimoku new {current_signal_1d} state stored for {pair}")

        # -------------------------------------------------
        # 1D neutral fallback message
        # -------------------------------------------------
        # Eğer 1D sinyal NEUTRAL'a döndüyse ama yukarıdaki lifecycle
        # özel mesajlarından biri üretilmediyse, eski genel mesajı gönder.
        # Bu nadir olur; çoğu NEUTRAL dönüşü confirmation failed veya invalidated olarak yakalanır.
        produced_ichimoku_lifecycle_message = any(
            "1d Ichimoku:" in msg for msg in notification_messages
        )

        if (
            has_previous_snapshot
            and changed_1d_lifecycle
            and current_signal_1d == "NEUTRAL"
            and not produced_ichimoku_lifecycle_message
        ):
            neutral_message = build_telegram_message(
                pair=pair,
                row_15m=latest_15m,
                signal_15m=snap_15m["signal"],
                row_1h=latest_1h,
                signal_1h=snap_1h["signal"],
                prev_signal_1h=prev_signal_1h,
                row_1d=latest_1d,
                signal_1d=snap_1d["signal"],
                prev_signal_1d=prev_signal_1d,
                ichi_details=ichi_details,
                candle_time_15m=snap_15m["last_open_time"],
                candle_time_1h=snap_1h["last_open_time"],
                candle_time_1d=snap_1d["last_open_time"],
                streamlit_app_url=STREAMLIT_APP_URL,
                triggered_15m=False,
                triggered_1h=False,
                triggered_1d=True,
            )
            notification_messages.append(neutral_message)

        if notification_messages:
            for message in notification_messages:
                send_telegram_message(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS)
            log(f"  └─ 📨 {len(notification_messages)} notification(s) sent for {pair}")
        else:
            log(f"  └─ Notification skipped for {pair} (no new event)")

    if not candle_rows:
        raise SystemExit("No candle rows prepared; nothing to upsert.")

    dedup_candles = {(r["pair"], r["timeframe"], r["open_time"]): r for r in candle_rows}
    dedup_snapshots = {(r["pair"], r["timeframe"]): r for r in snapshot_rows}

    log(f"⬆️ Upserting {len(dedup_candles)} candle rows into futures_candles")
    upsert_in_chunks(supabase, "futures_candles", dedup_candles.values(), on_conflict="pair,timeframe,open_time")

    log(f"⬆️ Upserting {len(dedup_snapshots)} snapshot rows into futures_signal_snapshots")
    upsert_in_chunks(supabase, "futures_signal_snapshots", dedup_snapshots.values(), on_conflict="pair,timeframe")

    log("✅ Process/publish stage complete")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
