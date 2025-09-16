#!/usr/bin/env python3
import os
import sys
import requests
import pandas as pd
from datetime import datetime
from supabase import create_client

# Env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SPARK_DAYS = int(os.getenv("SPARK_DAYS", "30"))
MIN_REQUIRED_POINTS = int(os.getenv("MIN_REQUIRED_POINTS", "10"))
WARN_ONLY = os.getenv("WARN_ONLY", "0") == "1"
BINANCE_API = "https://api.binance.com"
REQUEST_TIMEOUT = 20

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY", flush=True)
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Supabase fetchers ----------
def fetch_latest_snapshot() -> pd.DataFrame:
    r = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def fetch_history_paged() -> pd.DataFrame:
    """Pull full history with pagination (mirrors dashboard.py)."""
    page_size = 2000
    offset = 0
    rows = []
    while True:
        resp = (
            supabase
            .table("coin_price_daily")
            .select("coin,date,close")
            .order("coin", desc=False)
            .order("date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

# ---------- Binance check ----------
def get_trading_spot_symbols() -> set:
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        info = r.json()
        return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}
    except Exception as e:
        print(f"ℹ️ Could not fetch Binance symbols: {e}", flush=True)
        return set()

def main():
    print(f"▶ Sparkline diagnostics | SPARK_DAYS={SPARK_DAYS} | MIN_REQUIRED_POINTS={MIN_REQUIRED_POINTS} | WARN_ONLY={WARN_ONLY}")
    latest = fetch_latest_snapshot()
    hist = fetch_history_paged()

    if latest.empty:
        print("❌ coin_wma_latest is empty. Did the pipeline run?", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

    if hist.empty:
        print("❌ coin_price_daily is empty. No history to plot.", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

    spot_syms = get_trading_spot_symbols()  # for pair-existence hints

    latest_coins = sorted(latest["coin"].astype(str).unique().tolist())
    hist = hist.sort_values(["coin", "date"]).dropna(subset=["close"])

    # Precompute per-coin counts and last hist date
    counts = (
        hist.groupby("coin")
            .agg(n_rows=("date", "size"), last_hist=("date", "max"))
            .reset_index()
    )

    missing = []
    for coin in latest_coins:
        sub = hist.loc[hist["coin"] == coin]
        tail = sub.tail(SPARK_DAYS)
        n = len(tail.index)

        latest_date = latest.loc[latest["coin"] == coin, "date"].max()
        hist_last = tail["date"].max() if not tail.empty else None

        # helper facts
        total_rows = int(counts.loc[counts["coin"] == coin, "n_rows"].iloc[0]) if (counts["coin"] == coin).any() else 0
        last_hist_any = counts.loc[counts["coin"] == coin, "last_hist"].iloc[0] if (counts["coin"] == coin).any() else None
        pair = f"{coin.upper()}USDT"
        pair_exists = pair in spot_syms if spot_syms else None  # None if we couldn't fetch

        if n < MIN_REQUIRED_POINTS:
            missing.append((coin, n))
            reason_bits = []
            if total_rows == 0:
                reason_bits.append("no rows in coin_price_daily")
            else:
                reason_bits.append(f"{total_rows} total rows; last_hist={last_hist_any}")
            if pair_exists is True:
                reason_bits.append(f"pair {pair} exists on Binance")
            elif pair_exists is False:
                reason_bits.append(f"pair {pair} NOT on Binance")
            else:
                reason_bits.append("Binance pair unknown")
            reason = " | ".join(reason_bits)
            print(f"⚠️  {coin}: only {n} points (need ≥ {MIN_REQUIRED_POINTS}) — {reason}")
        else:
            print(f"✅ {coin}: {n} points | latest hist={hist_last} | latest snap={latest_date}")

    if missing:
        msg = f"❌ {len(missing)} coin(s) missing enough sparkline points: " + ", ".join(f"{c}({n})" for c, n in missing)
        print(msg, flush=True)
        sys.exit(0 if WARN_ONLY else 1)

    print("✅ All sparkline requirements satisfied.", flush=True)

if __name__ == "__main__":
    main()
