#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from crypto_futures_common import PAIR_LIST, TIMEFRAME_LIMITS, fetch_continuous_klines, log

OUTPUT_FILE = Path("binance_data.json")
GOOGLE_SHEETS_SYMBOLS_CSV_URL = os.getenv("GOOGLE_SHEETS_SYMBOLS_CSV_URL")


def _normalize_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().upper()
    return text in {"TRUE", "1", "YES", "Y"}


def _load_pairs_from_google_sheets_csv(url: str, retries: int = 3) -> list[str]:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            log(f"Trying Google Sheets symbol CSV (attempt {attempt}/{retries})")

            response = requests.get(url, timeout=20)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            expected_columns = {"symbol", "enabled"}
            actual_columns = {str(col).strip().lower() for col in df.columns}

            if not expected_columns.issubset(actual_columns):
                raise ValueError(
                    f"Google Sheets CSV must contain 'symbol' and 'enabled' columns. "
                    f"Found columns: {list(df.columns)}"
                )

            rename_map = {}
            for col in df.columns:
                col_lower = str(col).strip().lower()
                if col_lower == "symbol":
                    rename_map[col] = "symbol"
                elif col_lower == "enabled":
                    rename_map[col] = "enabled"

            df = df.rename(columns=rename_map)

            filtered = df[df["enabled"].apply(_normalize_enabled)].copy()

            symbols: list[str] = []
            for raw_symbol in filtered["symbol"].dropna().tolist():
                symbol = str(raw_symbol).strip().upper()
                if symbol:
                    symbols.append(symbol)

            deduped_symbols = list(dict.fromkeys(symbols))

            if not deduped_symbols:
                raise ValueError("Google Sheets CSV loaded successfully but no enabled symbols were found.")

            log(f"Loaded {len(deduped_symbols)} enabled symbols from Google Sheets CSV")
            log(f"Symbols from sheet: {', '.join(deduped_symbols)}")
            return deduped_symbols

        except Exception as exc:
            last_error = exc
            log(f"Google Sheets CSV read failed on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                sleep_seconds = attempt * 2
                log(f"Retrying Google Sheets CSV in {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)

    log(f"WARNING: Google Sheets CSV could not be used after {retries} attempts. Falling back to built-in PAIR_LIST.")
    if last_error is not None:
        log(f"Last Google Sheets CSV error: {last_error}")

    return list(PAIR_LIST)


def _get_active_pairs() -> list[str]:
    if not GOOGLE_SHEETS_SYMBOLS_CSV_URL:
        log("GOOGLE_SHEETS_SYMBOLS_CSV_URL is not set. Using built-in PAIR_LIST.")
        return list(PAIR_LIST)

    return _load_pairs_from_google_sheets_csv(GOOGLE_SHEETS_SYMBOLS_CSV_URL)


def run() -> None:
    active_pairs = _get_active_pairs()

    payload = {
        "pairs": {},
        "meta": {
            "source": "Binance continuousKlines",
            "contractType": "PERPETUAL",
            "timeframes": list(TIMEFRAME_LIMITS.keys()),
            "pair_source": "google_sheets_csv" if GOOGLE_SHEETS_SYMBOLS_CSV_URL else "built_in_pair_list",
            "pair_count": len(active_pairs),
        },
    }

    log(f"Active pair count: {len(active_pairs)}")
    log(f"Pairs to fetch: {', '.join(active_pairs)}")

    for pair in active_pairs:
        log(f"• Fetching {pair}")
        payload["pairs"][pair] = {}
        for timeframe, limit in TIMEFRAME_LIMITS.items():
            raw = fetch_continuous_klines(pair, timeframe, limit)
            payload["pairs"][pair][timeframe] = raw
            log(f"  └─ {timeframe}: fetched {len(raw)} raw candles")

    OUTPUT_FILE.write_text(json.dumps(payload), encoding="utf-8")
    log(f"✅ Binance raw data saved to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    run()
