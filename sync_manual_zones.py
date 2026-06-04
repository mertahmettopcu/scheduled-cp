#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests

from core_utils import create_supabase_client_from_env, log, upsert_in_chunks

GOOGLE_SHEETS_ZONES_CSV_URL = os.getenv("GOOGLE_SHEETS_ZONES_CSV_URL")

REQUIRED_COLUMNS = {"symbol", "zone_value", "active"}
OPTIONAL_COLUMNS = {"note"}

REMOTE_ZONE_LIST_RETRIES = 3
REMOTE_ZONE_LIST_TIMEOUT = 45
STALE_UPDATE_CHUNK_SIZE = 500


def _normalize_active(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().upper()
    return text in {"TRUE", "1", "YES", "Y"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _zone_key(symbol: object, zone_value: object) -> Tuple[str, float] | None:
    """
    Stable comparison key for Google Sheets zone rows and Supabase rows.

    Supabase numeric values can come back as int/float/string depending on the client.
    Rounding avoids tiny float representation differences while still preserving
    practical zone precision.
    """
    if symbol is None or zone_value is None or pd.isna(zone_value):
        return None

    symbol_text = str(symbol).strip().upper()
    if not symbol_text:
        return None

    try:
        value = round(float(zone_value), 8)
    except (TypeError, ValueError):
        return None

    return symbol_text, value


def _read_zones_csv(url: str) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, REMOTE_ZONE_LIST_RETRIES + 1):
        try:
            log(f"Trying remote manual zone list (attempt {attempt}/{REMOTE_ZONE_LIST_RETRIES})")

            response = requests.get(url, timeout=REMOTE_ZONE_LIST_TIMEOUT)
            response.raise_for_status()

            response.encoding = "utf-8"
            df = pd.read_csv(StringIO(response.text), encoding="utf-8")

            log(f"Remote manual zone list loaded with {len(df)} raw rows")
            return df

        except Exception as exc:
            last_error = exc
            log(f"Remote manual zone list read failed on attempt {attempt}/{REMOTE_ZONE_LIST_RETRIES}: {exc}")

            if attempt < REMOTE_ZONE_LIST_RETRIES:
                sleep_seconds = attempt * 2
                log(f"Retrying remote manual zone list in {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Remote manual zone list could not be loaded. Last error: {last_error}")


def _validate_and_prepare_zone_rows(df: pd.DataFrame) -> List[Dict]:
    if df.empty:
        raise ValueError("Manual zones CSV is empty. Supabase will not be updated.")

    original_columns = list(df.columns)
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Manual zones CSV is missing required columns: {sorted(missing_columns)}. "
            f"Found columns: {original_columns}"
        )

    work = df.copy()

    work["symbol"] = (
        work["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    work["active"] = work["active"].apply(_normalize_active)

    work["zone_value"] = pd.to_numeric(work["zone_value"], errors="coerce")

    if "note" not in work.columns:
        work["note"] = None

    invalid_rows = work[
        work["symbol"].eq("")
        | work["symbol"].isna()
        | work["zone_value"].isna()
        | (work["zone_value"] <= 0)
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Manual zones CSV contains invalid rows. "
            "Check empty symbol, non-numeric zone_value, or zone_value <= 0. "
            f"Invalid row indexes: {invalid_rows.index.tolist()}"
        )

    duplicate_mask = work.duplicated(subset=["symbol", "zone_value"], keep=False)
    if duplicate_mask.any():
        duplicates = work.loc[duplicate_mask, ["symbol", "zone_value"]].to_dict("records")
        raise ValueError(f"Duplicate manual zones found. Supabase will not be updated. Duplicates: {duplicates}")

    prepared_rows: List[Dict] = []
    updated_at = _now_iso()

    for symbol, group in work.groupby("symbol", sort=True):
        group_sorted = group.sort_values("zone_value", ascending=False).reset_index(drop=True)

        for idx, row in group_sorted.iterrows():
            prepared_rows.append(
                {
                    "symbol": row["symbol"],
                    "zone_value": float(row["zone_value"]),
                    "active": bool(row["active"]),
                    "sort_order": int(idx + 1),
                    "note": None if pd.isna(row.get("note")) else str(row.get("note")).strip(),
                    "source": "google_sheets",
                    "updated_at": updated_at,
                }
            )

    if not prepared_rows:
        raise ValueError("No valid manual zones found after validation. Supabase will not be updated.")

    return prepared_rows


def _fetch_google_sheet_zone_rows_from_supabase(supabase, symbols: List[str]) -> List[Dict]:
    if not symbols:
        return []

    all_rows: List[Dict] = []

    for symbol in symbols:
        resp = (
            supabase
            .table("manual_zones")
            .select("id,symbol,zone_value,active,source")
            .eq("symbol", symbol)
            .eq("source", "google_sheets")
            .execute()
        )
        all_rows.extend(resp.data or [])

    return all_rows


def _deactivate_stale_google_sheet_zones(
    supabase,
    *,
    rows: List[Dict],
    symbols: List[str],
) -> int:
    """
    Deactivate old google_sheets-sourced Supabase zone rows that no longer exist
    in the current Google Sheets CSV.

    This fixes the case where a zone value is changed in Sheets:
    - New value is upserted.
    - Old value is not in Sheets anymore.
    - Old value must be active=false so dashboard does not keep drawing it.

    Rows that still exist in Sheets are not touched here, even if their active value
    is FALSE, because the normal upsert already writes active=false for them.
    Rows with source != google_sheets are intentionally ignored.
    """
    current_keys: Set[Tuple[str, float]] = set()

    for row in rows:
        key = _zone_key(row.get("symbol"), row.get("zone_value"))
        if key is not None:
            current_keys.add(key)

    existing_rows = _fetch_google_sheet_zone_rows_from_supabase(supabase, symbols)

    stale_ids: List[int] = []
    stale_examples: List[str] = []

    for existing in existing_rows:
        is_active = _normalize_active(existing.get("active"))
        if not is_active:
            continue

        existing_key = _zone_key(existing.get("symbol"), existing.get("zone_value"))

        if existing_key is None:
            continue

        if existing_key not in current_keys:
            row_id = existing.get("id")
            if row_id is not None:
                stale_ids.append(int(row_id))
                if len(stale_examples) < 10:
                    stale_examples.append(f"{existing.get('symbol')} {existing.get('zone_value')} id={row_id}")

    if not stale_ids:
        log("No stale google_sheets manual zones to deactivate.")
        return 0

    updated_at = _now_iso()

    for i in range(0, len(stale_ids), STALE_UPDATE_CHUNK_SIZE):
        chunk = stale_ids[i:i + STALE_UPDATE_CHUNK_SIZE]
        (
            supabase
            .table("manual_zones")
            .update(
                {
                    "active": False,
                    "updated_at": updated_at,
                }
            )
            .in_("id", chunk)
            .execute()
        )

    log(f"Deactivated {len(stale_ids)} stale google_sheets manual zone rows.")
    if stale_examples:
        log("  └─ Examples: " + "; ".join(stale_examples))

    return len(stale_ids)


def run() -> None:
    if not GOOGLE_SHEETS_ZONES_CSV_URL:
        raise SystemExit("Missing GOOGLE_SHEETS_ZONES_CSV_URL")

    df = _read_zones_csv(GOOGLE_SHEETS_ZONES_CSV_URL)
    rows = _validate_and_prepare_zone_rows(df)

    symbols = sorted({row["symbol"] for row in rows})
    active_count = sum(1 for row in rows if row["active"])

    log(f"Prepared {len(rows)} manual zone rows for {len(symbols)} symbols")
    log(f"Active manual zone count: {active_count}")

    supabase = create_supabase_client_from_env()

    upsert_in_chunks(
        supabase=supabase,
        table_name="manual_zones",
        rows=rows,
        on_conflict="symbol,zone_value",
    )

    stale_count = _deactivate_stale_google_sheet_zones(
        supabase=supabase,
        rows=rows,
        symbols=symbols,
    )

    log(f"Stale manual zone rows deactivated: {stale_count}")
    log("✅ Manual zones synced to Supabase")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
