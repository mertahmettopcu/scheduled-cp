#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List

import pandas as pd
import requests

from core_utils import create_supabase_client_from_env, log, upsert_in_chunks

GOOGLE_SHEETS_ZONES_CSV_URL = os.getenv("GOOGLE_SHEETS_ZONES_CSV_URL")

REQUIRED_COLUMNS = {"symbol", "zone_value", "active"}
OPTIONAL_COLUMNS = {"note"}

REMOTE_ZONE_LIST_RETRIES = 3
REMOTE_ZONE_LIST_TIMEOUT = 45


def _normalize_active(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().upper()
    return text in {"TRUE", "1", "YES", "Y"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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

    log("✅ Manual zones synced to Supabase")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    run()
