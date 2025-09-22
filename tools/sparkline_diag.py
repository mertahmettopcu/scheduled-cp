#!/usr/bin/env python3
import os
import sys
import pandas as pd
from supabase import create_client

# ---- Early banner so we always see something ----
print("▶ Starting sparkline_diag.py bootstrap…", flush=True)
print("ℹ️ This diagnostic checks ONLY 30-day DAILY sparklines from Supabase.\n"
      "   The new 24h intraday chart is live-fetched from Binance and is NOT validated here.",
      flush=True)

# Env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SPARK_DAYS = int(os.getenv("SPARK_DAYS", "30"))
# Match dashboard tolerance (sparklines render with ≥3 usable points)
MIN_REQUIRED_POINTS = int(os.getenv("MIN_REQUIRED_POINTS", "3"))
# Default to warn-only so CI doesn't block deploy unless you override in workflow
WARN_ONLY = os.getenv("WARN_ONLY", "1") == "1"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY", flush=True)
    sys.exit(1 if not WARN_ONLY else 0)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def main():
    print(f"▶ Sparkline diagnostics | SPARK_DAYS={SPARK_DAYS} | "
          f"MIN_REQUIRED_POINTS={MIN_REQUIRED_POINTS} | WARN_ONLY={WARN_ONLY}", flush=True)

    latest = fetch_latest_snapshot()
    if latest.empty:
        print("❌ coin_wma_latest is empty. Did the pipeline run?", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

    hist = fetch_history_paged()
    if hist.empty:
        print("❌ coin_price_daily is empty. No daily history to plot.", flush=True)
        sys.exit(1 if not WARN_ONLY else 0)

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
        # count usable (non-NaN) points
        n = int(tail["close"].notna().sum())

        latest_date = latest.loc[latest["coin"] == coin, "date"].max()
        hist_last = tail["date"].max() if not tail.empty else None

        total_rows = int(counts.loc[counts["coin"] == coin, "n_rows"].iloc[0]) if (counts["coin"] == coin).any() else 0
        last_hist_any = counts.loc[counts["coin"] == coin, "last_hist"].iloc[0] if (counts["coin"] == coin).any() else None

        if n < MIN_REQUIRED_POINTS:
            missing.append((coin, n))
            reason = "no rows in coin_price_daily" if total_rows == 0 else f"{total_rows} total rows; last_hist={last_hist_any}"
            print(f"⚠️  {coin}: only {n} usable point(s) in last {SPARK_DAYS} days (need ≥ {MIN_REQUIRED_POINTS}) — {reason}", flush=True)
        else:
            print(f"✅ {coin}: {n} points | latest hist={hist_last} | latest snap={latest_date}", flush=True)

    if missing:
        msg = f"❌ {len(missing)} coin(s) missing enough sparkline points: " + ", ".join(f"{c}({n})" for c, n in missing)
        print(msg, flush=True)
        sys.exit(0 if WARN_ONLY else 1)

    print("✅ All daily sparkline requirements satisfied.", flush=True)

if __name__ == "__main__":
    # ensure unbuffered output even if CI forgets PYTHONUNBUFFERED
    try:
        import sys as _sys
        _sys.stdout.reconfigure(line_buffering=True)
        _sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
