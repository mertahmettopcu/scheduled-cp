#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from crypto_futures_common import PAIR_LIST, TIMEFRAME_LIMITS, fetch_continuous_klines, log

OUTPUT_FILE = Path("binance_data.json")


def run() -> None:
    payload = {
        "pairs": {},
        "meta": {
            "source": "Binance continuousKlines",
            "contractType": "PERPETUAL",
            "timeframes": list(TIMEFRAME_LIMITS.keys()),
        },
    }

    for pair in PAIR_LIST:
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
