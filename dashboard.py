import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # safe backend
import matplotlib.pyplot as plt
import requests
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

NOON_TZ = "Europe/Istanbul"
BINANCE_API = "https://api.binance.com"
REQUEST_TIMEOUT = 20

# -------------------- Supabase creds --------------------
def _read_supabase_creds():
    url = None
    key = None
    try:
        if "supabase" in st.secrets:
            url = st.secrets["supabase"].get("url")
            key = st.secrets["supabase"].get("key")
    except Exception:
        pass
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_KEY")
    if url: url = re.sub(r"\u00A0", " ", url).strip()
    if key: key = re.sub(r"\u00A0", " ", key).strip()
    return url, key

@st.cache_resource
def supabase_client():
    url, key = _read_supabase_creds()
    if not url or not key:
        st.error(
            "Supabase credentials not found.\n\n"
            "Add them in **Secrets** as:\n"
            "```toml\n[supabase]\nurl = \"https://<project>.supabase.co\"\nkey = \"<KEY>\"\n```\n"
            "or set env vars `SUPABASE_URL` and `SUPABASE_KEY`."
        )
        st.stop()
    return create_client(url, key)

supabase = supabase_client()

# -------------------- Data loader (snapshot only) --------------------
@st.cache_data(ttl=300)
def load_snapshot():
    snap = supabase.table("coin_wma_latest").select("*").execute()
    df_latest = pd.DataFrame(snap.data or [])
    return df_latest

# -------------------- Binance hourly fetch (no pre-check) --------------------
@st.cache_data(ttl=180)
def fetch_hourly_klines(symbol: str, hours: int = 48):
    """
    Fetch ~48h of 1h klines to ensure the latest Istanbul noon is inside.
    We will plot only the last 24 points.
    Returns (df, error_message). df has columns: open_time (UTC tz-aware), close (float).
    """
    pair = f"{symbol.upper()}USDT"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - hours * 3600 * 1000
    out = []
    curr = start_ms
    err = None

    while curr < end_ms:
        limit = min(1000, int((end_ms - curr) / (3600 * 1000)) + 1)
        try:
            r = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={"symbol": pair, "interval": "1h", "startTime": curr, "endTime": end_ms, "limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code in (429, 418, 451):
                time.sleep(0.5)
                continue
            if r.status_code >= 400:
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text
                err = f"Binance error {r.status_code}: {msg}"
                break
            data = r.json()
            if not data:
                break
            out.extend(data)
            last_open = data[-1][0]
            if last_open <= curr:
                break
            curr = last_open + 3600 * 1000
            time.sleep(0.06)
        except Exception as e:
            err = f"Request failed: {e}"
            break

    if not out:
        return None, err or "No data returned"

    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["close"]).sort_values("open_time").reset_index(drop=True)
    if df.empty:
        return None, "All closes were NaN/empty after parsing"
    return df, None

def plot_24h(dfh: pd.DataFrame, wma50: float | None, wma200: float | None, symbol: str):
    tail = dfh.tail(24).copy()
    if tail.empty:
        st.warning("Not enough hourly points to render the last 24h.")
        return

    x = np.arange(len(tail))
    y = tail["close"].to_numpy(dtype=float)

    # trendline on the 24 visible points
    mask = ~np.isnan(y)
    y_fit = None
    if mask.sum() >= 2:
        a, b = np.polyfit(x[mask], y[mask], 1)
        y_fit = a * x + b

    # Istanbul-noon marker within last 24h window
    local = tail["open_time"].dt.tz_convert(NOON_TZ)
    diffs = (local.dt.hour - 12).abs() + (local.dt.minute / 60.0)
    noon_idx = int(np.argmin(diffs.to_numpy())) if len(diffs) else None
    noon_time = tail["open_time"].iloc[noon_idx] if noon_idx is not None else None
    noon_price = tail["close"].iloc[noon_idx] if noon_idx is not None else None

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(tail["open_time"], y, linewidth=1.2, label=f"{symbol.upper()}/USDT (hourly)")
    if y_fit is not None:
        ax.plot(tail["open_time"], y_fit, linestyle="--", linewidth=1.2, label="Trend (24h)")

    # horizontal daily WMA refs
    if isinstance(wma50, (int, float)) and np.isfinite(wma50):
        ax.axhline(wma50, linestyle=":", linewidth=1.0, label="WMA 50 (daily)")
    if isinstance(wma200, (int, float)) and np.isfinite(wma200):
        ax.axhline(wma200, linestyle=":", linewidth=1.0, label="WMA 200 (daily)")

    if noon_time is not None and np.isfinite(noon_price):
        ax.scatter([noon_time], [noon_price], s=26, zorder=5)
        ax.annotate("Noon (TRT)", xy=(noon_time, noon_price),
                    xytext=(5, 8), textcoords="offset points", fontsize=8)

    ax.set_title(f"{symbol.upper()} — last 24h (hourly closes)", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, clear_figure=True)

# ==================== PAGE ====================
st.title("📈 Crypto WMA Dashboard")

df_latest = load_snapshot()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh.")
    st.stop()

# Normalize snapshot
df_latest.columns = [c.lower() for c in df_latest.columns]
df_latest["date"] = pd.to_datetime(df_latest["date"], errors="coerce")
for c in ("close","wma_50","wma_200"):
    if c in df_latest.columns:
        df_latest[c] = pd.to_numeric(df_latest[c], errors="coerce")

# Header
last_dt = df_latest["date"].max()
coins_shown = df_latest["coin"].nunique()
c1, c2 = st.columns(2)
c1.metric("Coins shown", f"{coins_shown}")
c2.metric("Last updated (UTC)", last_dt.strftime("%Y-%m-%d") if pd.notnull(last_dt) else "—")

# -------- Intraday 24h panel --------
st.subheader("Intraday (last 24h, hourly)")
coin_list = sorted(df_latest["coin"].astype(str).unique())
default_coin = "BTC" if "BTC" in coin_list else (coin_list[0] if coin_list else None)
selected = st.selectbox("Choose coin", coin_list, index=coin_list.index(default_coin) if default_coin else 0)

if selected:
    w50 = df_latest.loc[df_latest["coin"] == selected, "wma_50"].max()
    w200 = df_latest.loc[df_latest["coin"] == selected, "wma_200"].max()
    dfh, err = fetch_hourly_klines(selected, hours=48)
    if dfh is None:
        st.warning(f"Could not fetch hourly klines for {selected.upper()}USDT. {err or ''}".strip())
    else:
        plot_24h(dfh, w50, w200, selected)

# -------- Snapshot table (no daily sparklines) --------
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"

df_latest["status"] = df_latest.apply(status_badge, axis=1)

cols = ["coin","date","close","wma_50","wma_200","position","previous_position","status"]
st.subheader("Latest WMA snapshot per coin")
st.dataframe(
    df_latest[cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
    },
)
