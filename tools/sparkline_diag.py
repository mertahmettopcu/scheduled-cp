#!/usr/bin/env python3
import os
import sys
from datetime import datetime
import pandas as pd
from supabase import create_client

# Env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SPARK_DAYS = int(os.getenv("SPARK_DAYS", "30"))
MIN_REQUIRED_POINTS = int(os.getenv("MIN_REQUIRED_POINTS", "10"))
WARN_ONLY = os.getenv("WARN_ONLY", "0") == "1"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY", flush=True)
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_latest_snapshot() -> pd.DataFrame:
    r = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def fetch_history() -> pd.DataFrame:
    r = supabase.table("coin_price_daily").select("coin,date,close").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

def main():
    print(f"▶ Sparkline diagnostics | SPARK_DAYS={SPARK_DAYS} | MIN_REQUIRED_POINTS={MIN_REQUIRED_POINTS} | WARN_ONLY={WARN_ONLY}")
    latest = fetch_latest_snapshot()
    hist = fetch_history()

    if latest.empty:
        print("❌ coin_wma_latest is empty. Did the pipeline run?", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

    if hist.empty:
        print("❌ coin_wma is empty. No history to plot.", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

    latest_coins = sorted(latest["coin"].astype(str).unique().tolist())
    hist = hist.sort_values(["coin", "date"]).dropna(subset=["close"])

    missing = []
    for coin in latest_coins:
        sub = hist.loc[hist["coin"] == coin]
        # We expect at least MIN_REQUIRED_POINTS in the last SPARK_DAYS rows
        tail = sub.tail(SPARK_DAYS)
        n = len(tail.index)
        if n < MIN_REQUIRED_POINTS:
            missing.append((coin, n))
            print(f"⚠️  {coin}: only {n} points (need ≥ {MIN_REQUIRED_POINTS})")
        else:
            # Optional sanity: last point should be recent-ish compared to latest snapshot for that coin
            latest_date = latest.loc[latest["coin"] == coin, "date"].max()
            hist_last = tail["date"].max() if not tail.empty else None
            print(f"✅ {coin}: {n} points | latest hist={hist_last} | latest snap={latest_date}")

    if missing:
        msg = f"❌ {len(missing)} coin(s) missing enough sparkline points: " + ", ".join(f"{c}({n})" for c, n in missing)
        print(msg, flush=True)
        sys.exit(0 if WARN_ONLY else 1)

    print("✅ All sparkline requirements satisfied.", flush=True)

if __name__ == "__main__":
    main()
