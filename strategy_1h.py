#!/usr/bin/env python3
from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core_utils import (
    format_price,
    log,
    normalize_signal,
)

ZONE_BUFFER = 150.0
MOMENTUM_THRESHOLD = 0.35
SMA_BUFFER = 0.0
COUNTER_RSI_LONG_TO_SHORT_MAX = 30.0
COUNTER_RSI_SHORT_TO_LONG_MIN = 70.0
NORMAL_REVERSE_LONG_TO_SHORT_RSI4_MIN = 30.0
NORMAL_REVERSE_SHORT_TO_LONG_RSI4_MAX = 70.0
TP_AFTER_LONG_RSI4_MIN = 80.5
TP_AFTER_SHORT_RSI4_MAX = 20.0
SMA_LENGTHS = (65, 120, 168)
TP_NEAR_LEVELS = (
    ("L1", 25.0),
    ("L2", 15.0),
    ("L3", 7.5),
    ("L4", 5.0),
    ("L5", 1.0),
)

STATE_TABLE = "strategy_1h_states"
EVENT_TABLE = "strategy_1h_events"
TP_HISTORY_TABLE = "strategy_1h_tp_history"


def _iso(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _utc_offset_label(ts: pd.Timestamp) -> str:
    offset = ts.utcoffset()
    if offset is None:
        return "UTC"

    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)

    if minutes == 0:
        return f"UTC{sign}{hours}"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def _display_time_tr(value) -> str:
    if value is None or pd.isna(value):
        return "NA"

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    ts = ts.tz_convert("Europe/Istanbul")
    return f"{ts.strftime('%Y-%m-%d %H:%M')} ({_utc_offset_label(ts)} / İstanbul)"


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _zone_values(zones: pd.DataFrame) -> List[float]:
    if zones is None or zones.empty or "zone_value" not in zones.columns:
        return []
    return sorted(
        pd.to_numeric(zones["zone_value"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .astype(float)
        .tolist()
    )


def zone_context(price: float, zones: pd.DataFrame, direction: str) -> Optional[Dict]:
    values = _zone_values(zones)
    if not values or price is None or pd.isna(price):
        return None

    price = float(price)
    direction = normalize_signal(direction)

    exact_index = next(
        (
            i
            for i, value in enumerate(values)
            if math.isclose(price, value, rel_tol=0.0, abs_tol=1e-8)
        ),
        None,
    )

    if exact_index is not None:
        if direction == "LONG":
            if exact_index + 1 >= len(values):
                return None
            return {"lower": values[exact_index], "upper": values[exact_index + 1]}
        if direction == "SHORT":
            if exact_index - 1 < 0:
                return None
            return {"lower": values[exact_index - 1], "upper": values[exact_index]}

    lowers = [z for z in values if z < price]
    uppers = [z for z in values if z > price]
    if not lowers or not uppers:
        return None

    return {"lower": max(lowers), "upper": min(uppers)}


def next_zone(zone: float, zones: pd.DataFrame, direction: str) -> Optional[float]:
    values = _zone_values(zones)
    direction = normalize_signal(direction)
    if direction == "LONG":
        candidates = [z for z in values if z > float(zone)]
        return min(candidates) if candidates else None
    if direction == "SHORT":
        candidates = [z for z in values if z < float(zone)]
        return max(candidates) if candidates else None
    return None


def target_range(target_zone: float, zones: pd.DataFrame, direction: str) -> Optional[Dict]:
    values = _zone_values(zones)
    if target_zone is None:
        return None

    target_zone = float(target_zone)
    direction = normalize_signal(direction)

    if direction == "LONG":
        lowers = [z for z in values if z < target_zone]
        if not lowers:
            return None
        return {"lower": max(lowers), "upper": target_zone}

    if direction == "SHORT":
        uppers = [z for z in values if z > target_zone]
        if not uppers:
            return None
        return {"lower": target_zone, "upper": min(uppers)}

    return None


def zone_tp_trigger(target_zone: float, direction: str) -> float:
    direction = normalize_signal(direction)
    return (
        float(target_zone) - ZONE_BUFFER
        if direction == "LONG"
        else float(target_zone) + ZONE_BUFFER
    )


def in_zone_buffer(price: float, zone: float) -> bool:
    return float(zone) - ZONE_BUFFER <= float(price) <= float(zone) + ZONE_BUFFER


def target_zone_buffer_touched(row: pd.Series, target_zone: float, direction: str) -> bool:
    """Return True only if the candle actually touched the old target-zone buffer.

    This gates same-direction momentum target shifting after TP. Momentum alone
    can open a same-direction continuation, but it cannot advance the target to
    the next zone unless the old target-zone buffer has actually been touched.
    """
    if target_zone is None or pd.isna(target_zone):
        return False

    direction = normalize_signal(direction)
    trigger = zone_tp_trigger(float(target_zone), direction)

    if direction == "LONG":
        high = _safe_float(row.get("high"))
        return high is not None and high >= trigger

    if direction == "SHORT":
        low = _safe_float(row.get("low"))
        return low is not None and low <= trigger

    return False


def momentum_details(row: pd.Series, direction: str, zones: pd.DataFrame) -> Dict:
    context = zone_context(float(row["open"]), zones, direction)
    if context is None:
        return {
            "is_momentum": False,
            "body_size": None,
            "zone_distance": None,
            "ratio_pct": None,
            "reason": "ZONE_CONTEXT_UNAVAILABLE",
        }

    distance = float(context["upper"] - context["lower"])
    body = abs(float(row["close"]) - float(row["open"]))
    ratio = (body / distance) * 100.0 if distance > 0 else None
    direction_ok = (
        direction == "LONG" and float(row["close"]) > float(row["open"])
    ) or (
        direction == "SHORT" and float(row["close"]) < float(row["open"])
    )

    return {
        "is_momentum": bool(direction_ok and ratio is not None and ratio >= MOMENTUM_THRESHOLD * 100.0),
        "body_size": body,
        "zone_distance": distance,
        "ratio_pct": ratio,
        "lower_zone": context["lower"],
        "upper_zone": context["upper"],
        "reason": "OK",
    }


def _full_sma_order_ok(row: pd.Series, direction: str) -> bool:
    values = [row.get("sma65"), row.get("sma120"), row.get("sma168")]
    if any(pd.isna(v) for v in values):
        return False
    s65, s120, s168 = map(float, values)
    if direction == "LONG":
        return s65 < s120 < s168
    return s65 > s120 > s168


def choose_dynamic_tp(
    *,
    previous_closed_row: pd.Series,
    candle_open: float,
    direction: str,
    target_zone: float,
    zones: pd.DataFrame,
) -> Dict:
    direction = normalize_signal(direction)
    zone_trigger = zone_tp_trigger(target_zone, direction)

    if not _full_sma_order_ok(previous_closed_row, direction):
        return {
            "source": "ZONE",
            "raw_value": float(target_zone),
            "trigger": float(zone_trigger),
            "reason": "SMA_ORDER_NOT_VALID",
        }

    tr = target_range(target_zone, zones, direction)
    if tr is None:
        return {
            "source": "ZONE",
            "raw_value": float(target_zone),
            "trigger": float(zone_trigger),
            "reason": "TARGET_RANGE_UNAVAILABLE",
        }

    candidates: List[Dict] = []
    for length in SMA_LENGTHS:
        value = previous_closed_row.get(f"sma{length}")
        if value is None or pd.isna(value):
            continue
        value = float(value)

        inside_target_range = tr["lower"] <= value <= tr["upper"]
        price_side_ok = value > candle_open if direction == "LONG" else value < candle_open
        inside_zone_buffer = in_zone_buffer(value, target_zone)
        trigger = value - SMA_BUFFER if direction == "LONG" else value + SMA_BUFFER
        trigger_side_ok = trigger > candle_open if direction == "LONG" else trigger < candle_open

        if inside_target_range and price_side_ok and not inside_zone_buffer and trigger_side_ok:
            candidates.append(
                {
                    "source": f"SMA{length}",
                    "raw_value": value,
                    "trigger": float(trigger),
                    "reason": "NEAREST_VALID_SMA",
                }
            )

    if not candidates:
        return {
            "source": "ZONE",
            "raw_value": float(target_zone),
            "trigger": float(zone_trigger),
            "reason": "NO_VALID_SMA",
        }

    return min(candidates, key=lambda item: abs(float(item["trigger"]) - float(candle_open)))


def prepare_strategy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicators required by the live 1H engine without changing core_utils."""
    out = df.copy().sort_values("open_time").reset_index(drop=True)

    for length in (4, 16, 65, 120, 168):
        out[f"sma{length}"] = out["close"].rolling(length).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    def _rsi(period: int) -> pd.Series:
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    out["rsi4"] = _rsi(4)
    out["rsi14"] = _rsi(14)
    out["rsi52"] = _rsi(52)

    long_state = (
        (out["sma4"] > out["sma16"])
        & (out["rsi14"] > out["rsi52"])
        & (out["rsi14"] >= 50)
    )
    short_state = (
        (out["sma4"] < out["sma16"])
        & (out["rsi14"] < out["rsi52"])
        & (out["rsi14"] <= 50)
    )

    out["official_signal"] = "NEUTRAL"
    out.loc[long_state, "official_signal"] = "LONG"
    out.loc[short_state, "official_signal"] = "SHORT"
    return out


def get_last_directional_reference(df: pd.DataFrame) -> Dict:
    """Return the latest LONG/SHORT change while NEUTRAL does not erase direction."""
    if df.empty:
        return {"signal": "NEUTRAL", "signal_time": None, "row": None, "reason": "empty dataframe"}

    feat = prepare_strategy_df(df)
    last_directional = None
    latest = None

    for _, row in feat.iterrows():
        signal = normalize_signal(row.get("official_signal"))
        if signal not in {"LONG", "SHORT"}:
            continue
        if signal == last_directional:
            continue
        latest = (signal, row)
        last_directional = signal

    if latest is None:
        return {
            "signal": "NEUTRAL",
            "signal_time": None,
            "row": None,
            "reason": "no directional state-change event",
        }

    signal, row = latest
    return {
        "signal": signal,
        "signal_time": _iso(row["open_time"]),
        "row": row,
        "reason": "ok",
    }


def _default_state(pair: str) -> Dict:
    return {
        "pair": pair,
        "status": "FLAT",
        "trade_id": None,
        "direction": None,
        "entry_time": None,
        "entry_price": None,
        "entry_reason": None,
        "active_lower_zone": None,
        "active_upper_zone": None,
        "target_zone": None,
        "tp_source": None,
        "tp_raw_value": None,
        "tp_trigger": None,
        "tp_candle_open_time": None,
        "pending_tp_hit_time": None,
        "pending_tp_exit_price": None,
        "pending_tp_source": None,
        "pending_tp_raw_value": None,
        "pending_tp_trigger": None,
        "pending_closed_direction": None,
        "pending_old_target_zone": None,
        "last_processed_closed_open_time": None,
    }


def fetch_state(supabase, pair: str) -> Dict:
    resp = (
        supabase.table(STATE_TABLE)
        .select("*")
        .eq("pair", pair)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return _default_state(pair)
    state = _default_state(pair)
    state.update(rows[0])
    return state


def save_state(supabase, state: Dict) -> Dict:
    payload = dict(state)
    payload["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")
    resp = (
        supabase.table(STATE_TABLE)
        .upsert(payload, on_conflict="pair")
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else payload


def append_event(supabase, payload: Dict) -> None:
    row = dict(payload)
    row.setdefault("event_id", str(uuid.uuid4()))
    row.setdefault("created_at", pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"))
    supabase.table(EVENT_TABLE).insert(row).execute()


def append_tp_history(supabase, payload: Dict) -> None:
    row = dict(payload)
    row.setdefault("created_at", pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"))
    (
        supabase.table(TP_HISTORY_TABLE)
        .upsert(
            row,
            on_conflict="pair,trade_id,candle_open_time",
        )
        .execute()
    )


def _clear_position_fields(state: Dict) -> None:
    for key in (
        "trade_id",
        "direction",
        "entry_time",
        "entry_price",
        "entry_reason",
        "active_lower_zone",
        "active_upper_zone",
        "target_zone",
        "tp_source",
        "tp_raw_value",
        "tp_trigger",
        "tp_candle_open_time",
    ):
        state[key] = None


def _clear_pending_fields(state: Dict) -> None:
    for key in (
        "pending_tp_hit_time",
        "pending_tp_exit_price",
        "pending_tp_source",
        "pending_tp_raw_value",
        "pending_tp_trigger",
        "pending_closed_direction",
        "pending_old_target_zone",
    ):
        state[key] = None


def _open_position(
    *,
    supabase,
    state: Dict,
    pair: str,
    direction: str,
    entry_time,
    entry_price: float,
    reason: str,
    zones: pd.DataFrame,
    shifted_from_target: Optional[float] = None,
) -> Tuple[Dict, Optional[str]]:
    direction = normalize_signal(direction)
    context = zone_context(entry_price, zones, direction)
    if context is None:
        append_event(
            supabase,
            {
                "pair": pair,
                "event_type": "ZONE_CONTEXT_UNAVAILABLE",
                "event_time": _iso(entry_time),
                "direction": direction,
                "price": float(entry_price),
                "reason": "NEW_POSITION_BLOCKED",
                "details": {"entry_reason": reason},
            },
        )
        return state, build_zone_unavailable_message(pair, direction, entry_price, "New position was not opened")

    target = context["upper"] if direction == "LONG" else context["lower"]
    if shifted_from_target is not None:
        shifted = next_zone(float(shifted_from_target), zones, direction)
        if shifted is not None:
            target = shifted
        if direction == "LONG" and float(target) <= float(entry_price):
            target = context["upper"]
        if direction == "SHORT" and float(target) >= float(entry_price):
            target = context["lower"]

    trade_id = str(uuid.uuid4())
    state.update(
        {
            "status": "OPEN",
            "trade_id": trade_id,
            "direction": direction,
            "entry_time": _iso(entry_time),
            "entry_price": float(entry_price),
            "entry_reason": reason,
            "active_lower_zone": float(context["lower"]),
            "active_upper_zone": float(context["upper"]),
            "target_zone": float(target),
            "tp_source": None,
            "tp_raw_value": None,
            "tp_trigger": None,
            "tp_candle_open_time": None,
        }
    )
    _clear_pending_fields(state)

    append_event(
        supabase,
        {
            "pair": pair,
            "trade_id": trade_id,
            "event_type": "POSITION_OPEN",
            "event_time": _iso(entry_time),
            "direction": direction,
            "price": float(entry_price),
            "reason": reason,
            "active_lower_zone": float(context["lower"]),
            "active_upper_zone": float(context["upper"]),
            "target_zone": float(target),
        },
    )

    return state, build_position_open_message(
        pair=pair,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        reason=reason,
        active_lower=context["lower"],
        active_upper=context["upper"],
        target_zone=target,
    )


def _close_position(
    *,
    supabase,
    state: Dict,
    exit_time,
    exit_price: float,
    reason: str,
    details: Optional[Dict] = None,
) -> Dict:
    append_event(
        supabase,
        {
            "pair": state["pair"],
            "trade_id": state.get("trade_id"),
            "event_type": "POSITION_CLOSE",
            "event_time": _iso(exit_time),
            "direction": state.get("direction"),
            "price": float(exit_price),
            "reason": reason,
            "active_lower_zone": state.get("active_lower_zone"),
            "active_upper_zone": state.get("active_upper_zone"),
            "target_zone": state.get("target_zone"),
            "tp_source": state.get("tp_source"),
            "tp_trigger": state.get("tp_trigger"),
            "details": details or {},
        },
    )
    _clear_position_fields(state)
    state["status"] = "FLAT"
    return state


def _configure_tp_for_candle(
    *,
    supabase,
    state: Dict,
    previous_closed_row: pd.Series,
    candle_row: pd.Series,
    zones: pd.DataFrame,
) -> Tuple[Dict, Optional[str]]:
    if state.get("status") != "OPEN":
        return state, None

    context = zone_context(float(candle_row["open"]), zones, state["direction"])
    if context is None:
        previous_had_tp = state.get("tp_trigger") is not None
        state["active_lower_zone"] = None
        state["active_upper_zone"] = None
        state["tp_source"] = None
        state["tp_raw_value"] = None
        state["tp_trigger"] = None
        state["tp_candle_open_time"] = _iso(candle_row["open_time"])

        append_event(
            supabase,
            {
                "pair": state["pair"],
                "trade_id": state.get("trade_id"),
                "event_type": "ZONE_CONTEXT_UNAVAILABLE",
                "event_time": _iso(candle_row["open_time"]),
                "direction": state.get("direction"),
                "price": float(candle_row["open"]),
                "reason": "POSITION_KEPT_WITHOUT_TP",
            },
        )
        if previous_had_tp:
            return state, build_zone_unavailable_message(
                state["pair"], state["direction"], float(candle_row["open"]), "Open position kept; TP disabled"
            )
        return state, None

    state["active_lower_zone"] = float(context["lower"])
    state["active_upper_zone"] = float(context["upper"])

    target = state.get("target_zone")
    if target is None:
        target = context["upper"] if state["direction"] == "LONG" else context["lower"]
        state["target_zone"] = float(target)

    tp = choose_dynamic_tp(
        previous_closed_row=previous_closed_row,
        candle_open=float(candle_row["open"]),
        direction=state["direction"],
        target_zone=float(state["target_zone"]),
        zones=zones,
    )

    changed = (
        state.get("tp_source") != tp["source"]
        or state.get("tp_trigger") is None
        or not math.isclose(float(state["tp_trigger"]), float(tp["trigger"]), rel_tol=0.0, abs_tol=1e-8)
    )

    old_source = state.get("tp_source")
    old_trigger = state.get("tp_trigger")
    state.update(
        {
            "tp_source": tp["source"],
            "tp_raw_value": float(tp["raw_value"]),
            "tp_trigger": float(tp["trigger"]),
            "tp_candle_open_time": _iso(candle_row["open_time"]),
        }
    )

    append_tp_history(
        supabase,
        {
            "pair": state["pair"],
            "trade_id": state["trade_id"],
            "candle_open_time": _iso(candle_row["open_time"]),
            "direction": state["direction"],
            "tp_source": tp["source"],
            "tp_raw_value": float(tp["raw_value"]),
            "tp_trigger": float(tp["trigger"]),
            "target_zone": float(state["target_zone"]),
            "active_lower_zone": float(context["lower"]),
            "active_upper_zone": float(context["upper"]),
        },
    )

    if changed:
        append_event(
            supabase,
            {
                "pair": state["pair"],
                "trade_id": state["trade_id"],
                "event_type": "TP_UPDATE",
                "event_time": _iso(candle_row["open_time"]),
                "direction": state["direction"],
                "reason": tp["reason"],
                "target_zone": float(state["target_zone"]),
                "tp_source": tp["source"],
                "tp_raw_value": float(tp["raw_value"]),
                "tp_trigger": float(tp["trigger"]),
                "active_lower_zone": float(context["lower"]),
                "active_upper_zone": float(context["upper"]),
                "details": {
                    "previous_tp_source": old_source,
                    "previous_tp_trigger": old_trigger,
                },
            },
        )
        return state, build_tp_update_message(
            pair=state["pair"],
            direction=state["direction"],
            candle_time=candle_row["open_time"],
            previous_source=old_source,
            previous_trigger=old_trigger,
            source=tp["source"],
            raw_value=tp["raw_value"],
            trigger=tp["trigger"],
            target_zone=state["target_zone"],
            active_lower=context["lower"],
            active_upper=context["upper"],
            reason=tp["reason"],
        )

    return state, None


def _tp_hit(row: pd.Series, state: Dict) -> bool:
    trigger = state.get("tp_trigger")
    if trigger is None:
        return False
    if state["direction"] == "LONG":
        return float(row["high"]) >= float(trigger)
    return float(row["low"]) <= float(trigger)


def _mark_tp_pending(
    *,
    supabase,
    state: Dict,
    row: pd.Series,
) -> Tuple[Dict, str]:
    exit_price = float(state["tp_trigger"])
    if state["direction"] == "LONG" and float(row["open"]) >= exit_price:
        exit_price = float(row["open"])
    if state["direction"] == "SHORT" and float(row["open"]) <= exit_price:
        exit_price = float(row["open"])

    closed_direction = state["direction"]
    old_target = state["target_zone"]
    trade_id = state["trade_id"]
    hit_source = state.get("tp_source")
    hit_raw_value = state.get("tp_raw_value")
    hit_trigger = state.get("tp_trigger")

    append_event(
        supabase,
        {
            "pair": state["pair"],
            "trade_id": trade_id,
            "event_type": "TP_HIT",
            "event_time": _iso(row["open_time"]),
            "direction": closed_direction,
            "price": exit_price,
            "reason": f"TP_{hit_source}",
            "target_zone": old_target,
            "tp_source": hit_source,
            "tp_raw_value": hit_raw_value,
            "tp_trigger": hit_trigger,
            "active_lower_zone": state["active_lower_zone"],
            "active_upper_zone": state["active_upper_zone"],
            "details": {
                "candle_open_time": _iso(row["open_time"]),
                "waiting_for_candle_close": True,
            },
        },
    )

    message = build_tp_hit_message(
        pair=state["pair"],
        direction=closed_direction,
        trade_id=trade_id,
        candle_time=row["open_time"],
        source=hit_source,
        raw_value=hit_raw_value,
        trigger=hit_trigger,
        exit_price=exit_price,
        target_zone=old_target,
    )

    _clear_position_fields(state)
    state.update(
        {
            "status": "PENDING_TP_DECISION",
            "pending_tp_hit_time": _iso(row["open_time"]),
            "pending_tp_exit_price": exit_price,
            "pending_tp_source": hit_source,
            "pending_tp_raw_value": hit_raw_value,
            "pending_tp_trigger": hit_trigger,
            "pending_closed_direction": closed_direction,
            "pending_old_target_zone": old_target,
        }
    )
    return state, message


def _reset_pending_after_blocked_post_tp_entry(
    *,
    supabase,
    state: Dict,
    row: pd.Series,
    attempted_direction: str,
    entry_reason: str,
    official_signal: str,
) -> bool:
    """Clear stale TP-pending state when a post-TP entry attempt is blocked.

    _open_position keeps the incoming state unchanged when zone context is
    unavailable. During post-TP decision flow the incoming state is already
    PENDING_TP_DECISION, so a blocked entry must be finalized explicitly;
    otherwise later candles cannot fall back to the normal FLAT signal flow.
    """
    if state.get("status") != "PENDING_TP_DECISION":
        return False

    closed_direction = normalize_signal(state.get("pending_closed_direction"))
    old_target = state.get("pending_old_target_zone")
    rsi4 = _safe_float(row.get("rsi4"))

    append_event(
        supabase,
        {
            "pair": state["pair"],
            "event_type": "WAIT",
            "event_time": _iso(row["close_time"]),
            "direction": closed_direction if closed_direction in {"LONG", "SHORT"} else None,
            "price": float(row["close"]),
            "reason": "TP_AFTER_ENTRY_BLOCKED_FLAT_RESET",
            "target_zone": old_target,
            "details": {
                "blocked_entry_reason": entry_reason,
                "attempted_direction": normalize_signal(attempted_direction),
                "closed_direction": closed_direction,
                "old_target_zone": old_target,
                "official_signal": official_signal,
                "rsi4": rsi4,
                "note": "Post-TP entry was blocked, so pending TP decision was cleared and state returned to FLAT.",
            },
        },
    )
    _clear_pending_fields(state)
    state["status"] = "FLAT"
    return True


def _post_tp_decision(
    *,
    supabase,
    state: Dict,
    row: pd.Series,
    official_signal: str,
    zones: pd.DataFrame,
) -> Tuple[Dict, List[str]]:
    messages: List[str] = []
    direction = normalize_signal(state.get("pending_closed_direction"))
    old_target = state.get("pending_old_target_zone")

    if direction not in {"LONG", "SHORT"}:
        _clear_pending_fields(state)
        state["status"] = "FLAT"
        return state, messages

    close_in_buffer = old_target is not None and in_zone_buffer(float(row["close"]), float(old_target))
    same_momentum = momentum_details(row, direction, zones)

    if close_in_buffer:
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction=direction,
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="TP_AFTER_TARGET_BUFFER_CLOSE",
            zones=zones,
            shifted_from_target=old_target,
        )
        if msg:
            messages.append(msg)
        _reset_pending_after_blocked_post_tp_entry(
            supabase=supabase,
            state=state,
            row=row,
            attempted_direction=direction,
            entry_reason="TP_AFTER_TARGET_BUFFER_CLOSE",
            official_signal=official_signal,
        )
        return state, messages

    if same_momentum["is_momentum"]:
        zone_completed_for_shift = target_zone_buffer_touched(row, old_target, direction)
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction=direction,
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="TP_AFTER_SAME_DIRECTION_MOMENTUM",
            zones=zones,
            shifted_from_target=old_target if zone_completed_for_shift else None,
        )
        if msg:
            messages.append(msg)
        _reset_pending_after_blocked_post_tp_entry(
            supabase=supabase,
            state=state,
            row=row,
            attempted_direction=direction,
            entry_reason="TP_AFTER_SAME_DIRECTION_MOMENTUM",
            official_signal=official_signal,
        )
        return state, messages

    rsi4 = _safe_float(row.get("rsi4"))
    if direction == "LONG" and rsi4 is not None and rsi4 >= TP_AFTER_LONG_RSI4_MIN:
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction="SHORT",
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="TP_AFTER_RSI4_LONG_TO_SHORT",
            zones=zones,
        )
        if msg:
            messages.append(msg)
        _reset_pending_after_blocked_post_tp_entry(
            supabase=supabase,
            state=state,
            row=row,
            attempted_direction="SHORT",
            entry_reason="TP_AFTER_RSI4_LONG_TO_SHORT",
            official_signal=official_signal,
        )
        return state, messages

    if direction == "SHORT" and rsi4 is not None and rsi4 <= TP_AFTER_SHORT_RSI4_MAX:
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction="LONG",
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="TP_AFTER_RSI4_SHORT_TO_LONG",
            zones=zones,
        )
        if msg:
            messages.append(msg)
        _reset_pending_after_blocked_post_tp_entry(
            supabase=supabase,
            state=state,
            row=row,
            attempted_direction="LONG",
            entry_reason="TP_AFTER_RSI4_SHORT_TO_LONG",
            official_signal=official_signal,
        )
        return state, messages

    if official_signal in {"LONG", "SHORT"}:
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction=official_signal,
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="TP_AFTER_OFFICIAL_SIGNAL",
            zones=zones,
        )
        if msg:
            messages.append(msg)
        _reset_pending_after_blocked_post_tp_entry(
            supabase=supabase,
            state=state,
            row=row,
            attempted_direction=official_signal,
            entry_reason="TP_AFTER_OFFICIAL_SIGNAL",
            official_signal=official_signal,
        )
        return state, messages

    append_event(
        supabase,
        {
            "pair": state["pair"],
            "event_type": "WAIT",
            "event_time": _iso(row["close_time"]),
            "reason": "TP_AFTER_NO_BUFFER_MOMENTUM_RSI_OR_SIGNAL",
            "details": {
                "closed_direction": direction,
                "rsi4": rsi4,
                "official_signal": official_signal,
            },
        },
    )
    _clear_pending_fields(state)
    state["status"] = "FLAT"
    messages.append(
        build_wait_message(
            pair=state["pair"],
            candle_time=row["close_time"],
            reason="TP completed, but buffer close, momentum, RSI4 and official signal did not open a new position",
            rsi4=rsi4,
            official_signal=official_signal,
        )
    )
    return state, messages



def _tp_near_level_index(level: str) -> int:
    for idx, (name, _) in enumerate(TP_NEAR_LEVELS, start=1):
        if name == level:
            return idx
    return 0


def _tp_near_candidate(state: Dict, row: pd.Series) -> Optional[Dict]:
    if state.get("status") != "OPEN":
        return None

    direction = normalize_signal(state.get("direction"))
    if direction not in {"LONG", "SHORT"}:
        return None

    entry_price = _safe_float(state.get("entry_price"))
    trigger = _safe_float(state.get("tp_trigger"))
    if entry_price is None or trigger is None:
        return None

    total_distance = abs(float(trigger) - float(entry_price))
    if total_distance <= 0:
        return None

    if direction == "LONG":
        observed_price = _safe_float(row.get("high"))
        observed_label = "High"
        remaining = float(trigger) - float(observed_price) if observed_price is not None else None
    else:
        observed_price = _safe_float(row.get("low"))
        observed_label = "Low"
        remaining = float(observed_price) - float(trigger) if observed_price is not None else None

    if observed_price is None or remaining is None:
        return None

    # remaining <= 0 means TP is already touched or crossed; TP_HIT handles that case.
    if remaining <= 0:
        return None

    remaining_pct = (remaining / total_distance) * 100.0
    reached = [
        (idx, name, threshold)
        for idx, (name, threshold) in enumerate(TP_NEAR_LEVELS, start=1)
        if remaining_pct <= float(threshold)
    ]
    if not reached:
        return None

    level_index, level, threshold = max(reached, key=lambda item: item[0])
    progress_pct = max(0.0, min(100.0, 100.0 - remaining_pct))
    return {
        "level": level,
        "level_index": level_index,
        "threshold_remaining_pct": float(threshold),
        "remaining": float(remaining),
        "remaining_pct": float(remaining_pct),
        "progress_pct": float(progress_pct),
        "observed_price": float(observed_price),
        "observed_label": observed_label,
        "total_distance": float(total_distance),
        "trigger": float(trigger),
    }


def _max_sent_tp_near_level(supabase, state: Dict, trigger: float) -> int:
    try:
        resp = (
            supabase.table(EVENT_TABLE)
            .select("reason,tp_trigger,details")
            .eq("pair", state["pair"])
            .eq("trade_id", state.get("trade_id"))
            .eq("event_type", "TP_NEAR")
            .limit(100)
            .execute()
        )
    except Exception as exc:
        log(f"TP_NEAR duplicate check failed for {state.get('pair')}: {exc}")
        # Suppress alert if de-duplication cannot be checked; this avoids Telegram spam.
        return len(TP_NEAR_LEVELS)

    max_level = 0
    for item in resp.data or []:
        item_trigger = _safe_float(item.get("tp_trigger"))
        if item_trigger is None or not math.isclose(float(item_trigger), float(trigger), rel_tol=0.0, abs_tol=1e-8):
            continue

        details = item.get("details") or {}
        if isinstance(details, dict):
            level_index = details.get("level_index")
            if level_index is not None:
                try:
                    max_level = max(max_level, int(level_index))
                    continue
                except (TypeError, ValueError):
                    pass

        reason = str(item.get("reason") or "")
        if reason.startswith("TP_NEAR_"):
            max_level = max(max_level, _tp_near_level_index(reason.replace("TP_NEAR_", "")))

    return max_level


def _check_tp_near(
    *,
    supabase,
    state: Dict,
    row: pd.Series,
) -> Optional[str]:
    candidate = _tp_near_candidate(state, row)
    if candidate is None:
        return None

    max_sent_level = _max_sent_tp_near_level(supabase, state, candidate["trigger"])
    if candidate["level_index"] <= max_sent_level:
        return None

    reason = f"TP_NEAR_{candidate['level']}"
    append_event(
        supabase,
        {
            "pair": state["pair"],
            "trade_id": state.get("trade_id"),
            "event_type": "TP_NEAR",
            "event_time": _iso(row["open_time"]),
            "direction": state.get("direction"),
            "price": candidate["observed_price"],
            "reason": reason,
            "active_lower_zone": state.get("active_lower_zone"),
            "active_upper_zone": state.get("active_upper_zone"),
            "target_zone": state.get("target_zone"),
            "tp_source": state.get("tp_source"),
            "tp_raw_value": state.get("tp_raw_value"),
            "tp_trigger": state.get("tp_trigger"),
            "details": {
                "level": candidate["level"],
                "level_index": candidate["level_index"],
                "threshold_remaining_pct": candidate["threshold_remaining_pct"],
                "remaining": candidate["remaining"],
                "remaining_pct": candidate["remaining_pct"],
                "progress_pct": candidate["progress_pct"],
                "observed_label": candidate["observed_label"],
                "observed_price": candidate["observed_price"],
                "entry_price": state.get("entry_price"),
                "total_distance": candidate["total_distance"],
            },
        },
    )

    return build_tp_near_message(
        pair=state["pair"],
        direction=state["direction"],
        candle_time=row["open_time"],
        level=candidate["level"],
        source=state.get("tp_source"),
        trigger=state.get("tp_trigger"),
        observed_label=candidate["observed_label"],
        observed_price=candidate["observed_price"],
        remaining=candidate["remaining"],
        progress_pct=candidate["progress_pct"],
    )


def _counter_rsi_approved(position_direction: str, rsi4: Optional[float]) -> bool:
    if rsi4 is None:
        return False
    if position_direction == "LONG":
        return rsi4 <= COUNTER_RSI_LONG_TO_SHORT_MAX
    return rsi4 >= COUNTER_RSI_SHORT_TO_LONG_MIN


def _normal_reverse_rsi_approved(position_direction: str, rsi4: Optional[float]) -> bool:
    if rsi4 is None:
        return False
    if position_direction == "LONG":
        return rsi4 > NORMAL_REVERSE_LONG_TO_SHORT_RSI4_MIN
    return rsi4 < NORMAL_REVERSE_SHORT_TO_LONG_RSI4_MAX


def _normal_reverse_rsi_filter_text(position_direction: str) -> str:
    if position_direction == "LONG":
        return f"LONG→SHORT requires RSI4 > {NORMAL_REVERSE_LONG_TO_SHORT_RSI4_MIN:g}"
    return f"SHORT→LONG requires RSI4 < {NORMAL_REVERSE_SHORT_TO_LONG_RSI4_MAX:g}"


def _process_closed_candle(
    *,
    supabase,
    state: Dict,
    row: pd.Series,
    previous_closed_row: pd.Series,
    zones: pd.DataFrame,
) -> Tuple[Dict, List[str]]:
    messages: List[str] = []
    official_signal = normalize_signal(row.get("official_signal"))

    if state.get("status") == "PENDING_TP_DECISION":
        pending_time = state.get("pending_tp_hit_time")
        if pending_time and pd.Timestamp(pending_time) == pd.Timestamp(row["open_time"]):
            return _post_tp_decision(
                supabase=supabase,
                state=state,
                row=row,
                official_signal=official_signal,
                zones=zones,
            )

    if state.get("status") == "OPEN":
        state, tp_msg = _configure_tp_for_candle(
            supabase=supabase,
            state=state,
            previous_closed_row=previous_closed_row,
            candle_row=row,
            zones=zones,
        )
        if tp_msg:
            messages.append(tp_msg)

        if state.get("tp_trigger") is not None and _tp_hit(row, state):
            state, hit_msg = _mark_tp_pending(supabase=supabase, state=state, row=row)
            messages.append(hit_msg)
            state, post_messages = _post_tp_decision(
                supabase=supabase,
                state=state,
                row=row,
                official_signal=official_signal,
                zones=zones,
            )
            messages.extend(post_messages)
            return state, messages

        tp_near_msg = _check_tp_near(supabase=supabase, state=state, row=row)
        if tp_near_msg:
            messages.append(tp_near_msg)

        opposite = "SHORT" if state["direction"] == "LONG" else "LONG"
        if official_signal == opposite:
            counter = momentum_details(row, opposite, zones)
            rsi4 = _safe_float(row.get("rsi4"))

            if counter["is_momentum"]:
                if _counter_rsi_approved(state["direction"], rsi4):
                    old_direction = state["direction"]
                    old_trade_id = state["trade_id"]
                    state = _close_position(
                        supabase=supabase,
                        state=state,
                        exit_time=row["close_time"],
                        exit_price=float(row["close"]),
                        reason="COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_APPROVED",
                        details={
                            "rsi4": rsi4,
                            "counter_ratio_pct": counter.get("ratio_pct"),
                            "opposite_signal": opposite,
                        },
                    )
                    state, open_msg = _open_position(
                        supabase=supabase,
                        state=state,
                        pair=state["pair"],
                        direction=opposite,
                        entry_time=row["close_time"],
                        entry_price=float(row["close"]),
                        reason="COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_APPROVED",
                        zones=zones,
                    )
                    messages.append(
                        build_position_change_message(
                            pair=state["pair"],
                            old_direction=old_direction,
                            new_direction=opposite,
                            candle_time=row["close_time"],
                            price=float(row["close"]),
                            reason="COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_APPROVED",
                            details=[
                                f"Counter ratio: {counter.get('ratio_pct'):.2f}%",
                                f"RSI4: {rsi4:.2f}",
                                f"Filter: LONG→SHORT <= {COUNTER_RSI_LONG_TO_SHORT_MAX:g}; SHORT→LONG >= {COUNTER_RSI_SHORT_TO_LONG_MIN:g}",
                                f"Closed trade: {old_trade_id}",
                            ],
                        )
                    )
                    if open_msg:
                        # Position change message already covers the opening in detail.
                        pass
                    return state, messages

                append_event(
                    supabase,
                    {
                        "pair": state["pair"],
                        "trade_id": state["trade_id"],
                        "event_type": "POSITION_HELD",
                        "event_time": _iso(row["close_time"]),
                        "direction": state["direction"],
                        "price": float(row["close"]),
                        "reason": "COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_REJECTED",
                        "details": {
                            "rsi4": rsi4,
                            "counter_ratio_pct": counter.get("ratio_pct"),
                            "official_signal": official_signal,
                        },
                    },
                )
                messages.append(
                    build_position_held_message(
                        pair=state["pair"],
                        direction=state["direction"],
                        candle_time=row["close_time"],
                        price=float(row["close"]),
                        counter_ratio=counter.get("ratio_pct"),
                        rsi4=rsi4,
                        opposite_signal=official_signal,
                    )
                )
                return state, messages

            if not _normal_reverse_rsi_approved(state["direction"], rsi4):
                append_event(
                    supabase,
                    {
                        "pair": state["pair"],
                        "trade_id": state["trade_id"],
                        "event_type": "POSITION_HELD",
                        "event_time": _iso(row["close_time"]),
                        "direction": state["direction"],
                        "price": float(row["close"]),
                        "reason": "NORMAL_REVERSE_SIGNAL_RSI_REJECTED",
                        "details": {
                            "rsi4": rsi4,
                            "official_signal": official_signal,
                            "filter": _normal_reverse_rsi_filter_text(state["direction"]),
                        },
                    },
                )
                return state, messages

            old_direction = state["direction"]
            old_trade_id = state["trade_id"]
            filter_text = _normal_reverse_rsi_filter_text(old_direction)
            state = _close_position(
                supabase=supabase,
                state=state,
                exit_time=row["close_time"],
                exit_price=float(row["close"]),
                reason="NORMAL_REVERSE_SIGNAL_RSI_APPROVED",
                details={
                    "official_signal": official_signal,
                    "rsi4": rsi4,
                    "filter": filter_text,
                },
            )
            state, _ = _open_position(
                supabase=supabase,
                state=state,
                pair=state["pair"],
                direction=opposite,
                entry_time=row["close_time"],
                entry_price=float(row["close"]),
                reason="NORMAL_REVERSE_SIGNAL_RSI_APPROVED",
                zones=zones,
            )
            messages.append(
                build_position_change_message(
                    pair=state["pair"],
                    old_direction=old_direction,
                    new_direction=opposite,
                    candle_time=row["close_time"],
                    price=float(row["close"]),
                    reason="NORMAL_REVERSE_SIGNAL_RSI_APPROVED",
                    details=[
                        f"Official signal: {official_signal}",
                        f"RSI4: {'NA' if rsi4 is None else f'{rsi4:.2f}'}",
                        f"Filter: {filter_text}",
                        f"Closed trade: {old_trade_id}",
                    ],
                )
            )
            return state, messages

    if state.get("status") == "FLAT" and official_signal in {"LONG", "SHORT"}:
        state, msg = _open_position(
            supabase=supabase,
            state=state,
            pair=state["pair"],
            direction=official_signal,
            entry_time=row["close_time"],
            entry_price=float(row["close"]),
            reason="OFFICIAL_SIGNAL_WHILE_FLAT",
            zones=zones,
        )
        if msg:
            messages.append(msg)

    return state, messages


def run_live_1h_strategy(
    *,
    supabase,
    pair: str,
    closed_1h_df: pd.DataFrame,
    open_1h_row: Optional[pd.Series],
    zones: pd.DataFrame,
) -> List[str]:
    messages: List[str] = []

    if closed_1h_df.empty:
        return messages

    work = prepare_strategy_df(closed_1h_df)
    state = fetch_state(supabase, pair)

    last_processed = state.get("last_processed_closed_open_time")
    start_index = 1
    if last_processed:
        last_ts = pd.Timestamp(last_processed)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        candidates = work.index[work["open_time"] > last_ts].tolist()
        start_index = candidates[0] if candidates else len(work)
    else:
        # First activation: process only latest closed candle, do not replay full history.
        start_index = max(1, len(work) - 1)

    for i in range(start_index, len(work)):
        row = work.iloc[i]
        prev = work.iloc[i - 1]
        state, new_messages = _process_closed_candle(
            supabase=supabase,
            state=state,
            row=row,
            previous_closed_row=prev,
            zones=zones,
        )
        messages.extend(new_messages)
        state["last_processed_closed_open_time"] = _iso(row["open_time"])
        state = save_state(supabase, state)

    if open_1h_row is not None and state.get("status") == "OPEN":
        latest_closed = work.iloc[-1]
        state, tp_msg = _configure_tp_for_candle(
            supabase=supabase,
            state=state,
            previous_closed_row=latest_closed,
            candle_row=open_1h_row,
            zones=zones,
        )
        if tp_msg:
            messages.append(tp_msg)

        if state.get("tp_trigger") is not None and _tp_hit(open_1h_row, state):
            state, hit_msg = _mark_tp_pending(
                supabase=supabase,
                state=state,
                row=open_1h_row,
            )
            messages.append(hit_msg)
        elif state.get("status") == "OPEN":
            tp_near_msg = _check_tp_near(supabase=supabase, state=state, row=open_1h_row)
            if tp_near_msg:
                messages.append(tp_near_msg)

        state = save_state(supabase, state)

    return messages


def _reason_description(reason: str) -> str:
    descriptions = {
        "OFFICIAL_SIGNAL_WHILE_FLAT": "Resmî sinyal ile açılış",
        "NORMAL_REVERSE_SIGNAL": "Normal ters sinyal",
        "NORMAL_REVERSE_SIGNAL_RSI_APPROVED": "Normal ters sinyal + RSI4 onay",
        "NORMAL_REVERSE_SIGNAL_RSI_REJECTED": "Normal ters sinyal + RSI4 red",
        "COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_APPROVED": "Counter-momentum + RSI4 onay",
        "COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_REJECTED": "Counter-momentum + RSI4 red",
        "TP_AFTER_TARGET_BUFFER_CLOSE": "TP sonrası target buffer",
        "TP_AFTER_SAME_DIRECTION_MOMENTUM": "TP sonrası aynı yön momentum",
        "TP_AFTER_RSI4_LONG_TO_SHORT": "LONG TP sonrası RSI4 → SHORT",
        "TP_AFTER_RSI4_SHORT_TO_LONG": "SHORT TP sonrası RSI4 → LONG",
        "TP_AFTER_OFFICIAL_SIGNAL": "TP sonrası resmî sinyal",
        "SMA_ORDER_NOT_VALID": "SMA sıralaması uygun değil",
        "NO_VALID_SMA": "Geçerli SMA TP yok",
        "NEAREST_VALID_SMA": "En yakın geçerli SMA TP",
        "TARGET_RANGE_UNAVAILABLE": "Target aralığı yok",
        "ZONE_CONTEXT_UNAVAILABLE": "Zone bağlamı yok",
        "TP_NEAR_L1": "TP yakın L1",
        "TP_NEAR_L2": "TP yakın L2",
        "TP_NEAR_L3": "TP yakın L3",
        "TP_NEAR_L4": "TP yakın L4",
        "TP_NEAR_L5": "TP yakın L5",
    }
    text = descriptions.get(str(reason or ""))
    if text:
        return text
    return str(reason or "").replace("_", " ").title()


def _compact_details(details: List[str], max_lines: int = 3) -> List[str]:
    cleaned: List[str] = []
    for item in details or []:
        text = str(item)
        if text.startswith("Closed trade:"):
            continue
        if text.startswith("Filter:"):
            text = text.replace("LONG→SHORT requires", "L→S şartı:").replace("SHORT→LONG requires", "S→L şartı:")
        if text.startswith("Official signal:"):
            text = text.replace("Official signal:", "Sinyal:")
        cleaned.append(text)
    return cleaned[:max_lines]


def build_position_open_message(
    *,
    pair: str,
    direction: str,
    entry_time,
    entry_price: float,
    reason: str,
    active_lower: float,
    active_upper: float,
    target_zone: float,
) -> str:
    return "\n".join(
        [
            f"🟢 {pair} 1H OPEN {direction}",
            f"Entry: {format_price(entry_price)}",
            f"Reason: {_reason_description(reason)}",
            f"Zone: {format_price(active_lower)} → {format_price(active_upper)}",
            f"Target: {format_price(target_zone)}",
            f"Time TR: {_display_time_tr(entry_time)}",
        ]
    )


def build_position_change_message(
    *,
    pair: str,
    old_direction: str,
    new_direction: str,
    candle_time,
    price: float,
    reason: str,
    details: List[str],
) -> str:
    short_old = str(old_direction or "?")[:1]
    short_new = str(new_direction or "?")[:1]
    return "\n".join(
        [
            f"🟠 {pair} 1H SWITCH {short_old}→{short_new}",
            f"Price: {format_price(price)}",
            f"Reason: {_reason_description(reason)}",
            *_compact_details(details, max_lines=3),
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )


def build_position_held_message(
    *,
    pair: str,
    direction: str,
    candle_time,
    price: float,
    counter_ratio: Optional[float],
    rsi4: Optional[float],
    opposite_signal: str,
) -> str:
    ratio_text = "NA" if counter_ratio is None else f"{counter_ratio:.2f}%"
    rsi_text = "NA" if rsi4 is None else f"{rsi4:.2f}"
    return "\n".join(
        [
            f"🟣 {pair} 1H HELD {direction}",
            f"Close: {format_price(price)}",
            "Reason: Counter-momentum + RSI4 red",
            f"Opposite: {opposite_signal}",
            f"Counter: {ratio_text}",
            f"RSI4: {rsi_text}",
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )


def build_tp_update_message(
    *,
    pair: str,
    direction: str,
    candle_time,
    previous_source,
    previous_trigger,
    source: str,
    raw_value: float,
    trigger: float,
    target_zone: float,
    active_lower: float,
    active_upper: float,
    reason: str,
) -> str:
    return "\n".join(
        [
            f"🟡 {pair} 1H TP SET {direction}",
            f"TP: {source} @ {format_price(trigger)}",
            f"Target: {format_price(target_zone)}",
            f"Reason: {_reason_description(reason)}",
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )


def build_tp_hit_message(
    *,
    pair: str,
    direction: str,
    trade_id: str,
    candle_time,
    source: str,
    raw_value: float,
    trigger: float,
    exit_price: float,
    target_zone: float,
) -> str:
    return "\n".join(
        [
            f"⭐ {pair} 1H TP HIT {direction}",
            f"Exit: {format_price(exit_price)}",
            f"TP: {source} @ {format_price(trigger)}",
            f"Target: {format_price(target_zone)}",
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )



def build_tp_near_message(
    *,
    pair: str,
    direction: str,
    candle_time,
    level: str,
    source,
    trigger,
    observed_label: str,
    observed_price: float,
    remaining: float,
    progress_pct: float,
) -> str:
    source_text = source or "TP"
    return "\n".join(
        [
            f"🔔 {pair} 1H TP NEAR {level} {direction}",
            f"TP: {source_text} @ {format_price(trigger)}",
            f"{observed_label}: {format_price(observed_price)}",
            f"Remaining: {format_price(remaining)}",
            f"Progress: {progress_pct:.1f}%",
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )


def build_wait_message(
    *,
    pair: str,
    candle_time,
    reason: str,
    rsi4: Optional[float],
    official_signal: str,
) -> str:
    return "\n".join(
        [
            f"⚪ {pair} 1H WAIT",
            f"Reason: {reason}",
            f"RSI4: {'NA' if rsi4 is None else f'{rsi4:.2f}'}",
            f"Signal: {official_signal}",
            f"Time TR: {_display_time_tr(candle_time)}",
        ]
    )


def build_zone_unavailable_message(pair: str, direction: str, price: float, action: str) -> str:
    return "\n".join(
        [
            f"⚠️ {pair} 1H ZONE ERROR",
            f"Direction: {direction}",
            f"Price: {format_price(price)}",
            f"Action: {action}",
            "Zone update required.",
        ]
    )
