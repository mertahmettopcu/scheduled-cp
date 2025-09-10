#!/usr/bin/env python3
"""
sparkline_diag.py

Print per-coin diagnostics for the dashboard sparkline (last 30 days).
This runs in CI and writes to STDOUT so you can see it in the Actions logs.

Env:
  SUPABASE_URL, SUPABASE_KEY
"""

import os
import sys
import numpy as np
import pandas as pd
from supabase import create_client

SPARK_DAYS = 30

def main():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY is missing in environment.", file=sys.stderr)
        sys.exit(2)

    supabase = create_client(url, key)

    # Pull only what we need
    resp = supabase.table("coin_wma").select("coin,date,close").execute()
    rows = resp.data or []
    if not rows:
        print("No rows in coin_wma. Nothing to diagnose.")
        return

    df = pd.DataFrame(rows)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["coin", "date"]).sort_values(["coin", "date"])

    print("\n===== Sparkline diagnostics (last 30d) =====")
    bad = 0

    for coin in df["coin"].dropna().unique():
        g = df[df["coin"] == coin]
        hist_rows = int(len(g))
        last_hist_date = g["date"].max()

        s = g[["date", "close"]].dropna().sort_values("date")
        y = pd.to_numeric(s["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        recent = y.tail(SPARK_DAYS)
        recent = recent.interpolate(limit_direction="both").ffill().bfill()
        finite = np.isfinite(recent.to_numpy(dtype=float))
        finite_recent = int(finite.sum())
        recent_rows = int(len(recent))

        if finite_recent == 0:
            status = "NO_DATA"
            reason = "no finite closes in last 30d"
            bad += 1
        else:
            status = "OK"
            reason = ""

        print(f"{coin:6s} | hist_rows={hist_rows:4d} "
              f"| last={last_hist_date.date() if pd.notnull(last_hist_date) else '—'} "
              f"| recent={recent_rows:2d} | finite_recent={finite_recent:2d} "
              f"| {status} {('('+reason+')') if reason else ''}")

    if bad:
        print(f"\nSummary: {bad} coin(s) have NO_DATA in last 30d.")
    else:
        print("\nSummary: All coins have finite data for the last 30d.")

if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
