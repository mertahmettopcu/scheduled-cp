#!/usr/bin/env python3
"""
crypto_pipeline_with_telegram.py
- Fetch top 30 coins from CoinGecko
- Download hourly klines from Binance, pick candle closest to 12:00 Europe/Istanbul each day
- Compute WMA(50) and WMA(200)
- Classify position and previous_position
- Upsert into Supabase (coin_wma table)
- Send Telegram alerts on position change (latest vs previous)

Env:
  SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client, Client

# -------------------------- Config (via environment variables) --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Prefer anon for read-only; service role only on trusted servers
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY env vars. See guide.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------- Telegram helper --------------------------------------------
def send_telegram_alert(message: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram not configured; skipping alert.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            print("Telegram API returned", resp.status_code, resp.text)
    except Exception as e:
        print("Failed to send Telegram message:", e)

# -------------------------- CoinGecko top N --------------------------------------------
def get_top_symbols_for_headroom(limit: int = 100) -> list[str]:
    """
    Ask CoinGecko for many (100) coins to have slack, then we can keep the first 30 that
    actually trade against USDT on Binance.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return [item["symbol"].upper() for item in data]

# -------------------------- Binance helpers --------------------------------------------
BINANCE_API = "https://api.binance.com"
BINANCE_KLINES_URL = f"{BINANCE_API}/api/v3/klines"

def get_binance_spot_symbols() -> set[str]:
    r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    info = r.json()
    return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

def map_to_valid_pair(bases: list[str], quote: str = "USDT") -> list[str]:
    """
