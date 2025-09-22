import os
import re
import io
import time
import base64
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # safe backend
import matplotlib.pyplot as plt
import requests
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

SPARK_DAYS = 30
MIN_SPARK_POINTS = int(os.getenv("MIN_SPARK_POINTS", "3"))
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

# -------------------- Data loader (snapshot + daily history) --------------------
@st.cache_data(ttl=300)
def load_data():
    snap = supabase.table("coin_wma_latest").select("*").execute()
    df_latest = pd.DataFrame(snap.data or [])

    # paged fetch for coin_price_daily (sparklines)
    page_size = 2000
    offset = 0
    hist_rows = []
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
        hist_rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break

    df_hist = pd.DataFrame(hist_rows)
    return df_latest, df_hist

# ==================== NEW: Intraday (hourly) 24h panel ====================
@st.cache_data(ttl=180)
def _binance_spot_symbols() -> set:
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        info = r.json()
        return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}
    except Exception:
        return set()

def _pair_exists(symbol: str) -> bool:
    syms = _binance_spot_symbols()
    return f"{symbol.upper()}USDT" in syms

@st.cache_data(ttl=180)
def _fetch_hourly_klines(symbol: str, hours: int = 48) -> pd.DataFrame | None:
    """
    Fetch ~48h of hourly klines (UTC) to ensure we capture the latest Istanbul noon;
    we will plot only the last 24 points.
    """
    pair = f"{symbol.upper()}USDT"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - hours * 3600 * 1000
    out = []
    curr = start_ms
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
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            out.extend(data)
            last_open = data[-1][0]
            if last_open <= curr:
                break
            curr = last_open + 3600 * 1000
            time.sleep(0.06)
        except Exception:
            break

    if not out:
        return None

    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["close"]).sort_values("open_time").reset_index(drop=True)
    return df

def _plot_24h_hourly_with_refs(dfh: pd.DataFrame, wma50: float | None, wma200: float | None, symbol: str):
    """
    Plot last 24 hourly closes with:
      - price line,
      - linear trendline,
      - noon(Istanbul) marker if present,
      - horizontal lines for daily WMA-50 and WMA-200 (if provided).
    """
    tail = dfh.tail(24).copy()
    if tail.empty:
        st.warning("Not enough hourly points to render the last 24h.")
        return

    x = np.arange(len(tail))
    y = tail["close"].to_numpy(dtype=float)

    # trendline on the 24 visible points
    mask = ~np.isnan(y)
    if mask.sum() >= 2:
        a, b = np.polyfit(x[mask], y[mask], 1)
        y_fit = a * x + b
    else:
        y_fit = None

    # find Istanbul-noon candle within 24h
    local = tail["open_time"].dt.tz_convert(NOON_TZ)
    # choose the row closest to 12:00 local on its day
    diffs = (local.dt.hour - 12).abs() + (local.dt.minute / 60.0)
    noon_idx = int(np.argmin(diffs.to_numpy())) if len(diffs) else None
    noon_time = tail["open_time"].iloc[noon_idx] if noon_idx is not None else None
    noon_price = tail["close"].iloc[noon_idx] if noon_idx is not None else None

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(tail["open_time"], y, linewidth=1.2, label=f"{symbol.upper()}/USDT (hourly)")
    if y_fit is not None:
        ax.plot(tail["open_time"], y_fit, linestyle="--", linewidth=1.2, label="Trend (24h)")

    # horizontal reference lines (DAILY WMAs)
    if isinstance(wma50, (int, float)) and np.isfinite(wma50):
        ax.axhline(wma50, linestyle=":", linewidth=1.0, label="WMA 50 (daily)")
    if isinstance(wma200, (int, float)) and np.isfinite(wma200):
        ax.axhline(wma200, linestyle=":", linewidth=1.0, label="WMA 200 (daily)")

    # noon marker if within view
    if noon_time is not None and np.isfinite(noon_price):
        ax.scatter([tail["open_time"].iloc[noon_idx]], [noon_price], s=24, zorder=5)
        ax.annotate("Noon (TRT)", xy=(tail["open_time"].iloc[noon_idx], noon_price),
                    xytext=(5, 8), textcoords="offset points", fontsize=8)

    ax.set_title(f"{symbol.upper()} — last 24h (hourly closes)", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    st.pyplot(fig, clear_figure=True)

# -------------------- Sparkline builder (unchanged except robustness) --------------------
def _spark_with_trend_png(y: np.ndarray) -> str | None:
    if y is None:
        return None
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    x_masked = x[mask]
    y_masked = y[mask]
    if len(y_masked) < MIN_SPARK_POINTS:
        return None

    a, b = np.polyfit(x_masked, y_masked, 1)
    y_fit = a * x + b

    fig, ax = plt.subplots(figsize=(2.4, 0.6), dpi=150)
    ax.plot(x, y, linewidth=1.0)
    ax.plot(x, y_fit, linestyle="--", linewidth=1.2)
    ax.axis("off")
    ymin = np.nanmin(y_masked)
    ymax = np.nanmax(y_masked)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@st.cache_data(ttl=300)
def build_trend_images(df_hist: pd.DataFrame, days: int) -> pd.DataFrame:
    if df_hist.empty:
        return pd.DataFrame(columns=["coin", "trend_img", "n_points"])

    df = df_hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    tailn = (
        df.sort_values(["coin", "date"])
          .groupby("coin", as_index=False, group_keys=False)
          .tail(days)
    )

    images = []
    for coin, sub in tailn.groupby("coin"):
        y = sub["close"].astype(float).to_numpy()
        n_points = int(np.sum(~np.isnan(y)))
        img = _spark_with_trend_png(y) if n_points >= MIN_SPARK_POINTS else None
        images.append({"coin": coin, "trend_img": img, "n_points": n_points})
    return pd.DataFrame(images)

# ==================== PAGE LAYOUT ====================
st.title("📈 Crypto WMA Dashboard")

df_latest, df_hist = load_data()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh.")
    st.stop()

# Normalize latest snapshot
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

# -------- NEW intraday 24h panel --------
st.subheader("Intraday (last 24h, hourly)")
coin_list = sorted(df_latest["coin"].astype(str).unique())
default_coin = "BTC" if "BTC" in coin_list else (coin_list[0] if coin_list else None)
selected = st.selectbox("Choose coin", coin_list, index=coin_list.index(default_coin) if default_coin else 0)

if selected:
    w50 = df_latest.loc[df_latest["coin"] == selected, "wma_50"].max()
    w200 = df_latest.loc[df_latest["coin"] == selected, "wma_200"].max()

    pair = f"{selected.upper()}USDT"
    if not _pair_exists(selected):
        st.warning(f"Binance spot pair **{pair}** not found.")
    else:
        dfh = _fetch_hourly_klines(selected, hours=48)
        if dfh is None or dfh.empty:
            st.warning(f"No hourly data returned for **{pair}**.")
        else:
            _plot_24h_hourly_with_refs(dfh, w50, w200, selected)

# -------- Build tiny sparkline images and merge onto snapshot (existing) --------
trend_imgs = build_trend_images(df_hist, SPARK_DAYS)
df_latest = df_latest.merge(trend_imgs, on="coin", how="left")

# Coverage expander
with st.expander("Sparkline data coverage (last 30 rows)", expanded=False):
    coverage = (
        df_latest[["coin", "n_points"]]
        .sort_values("coin")
        .rename(columns={"n_points": f"usable_points (≥{MIN_SPARK_POINTS})"})
    )
    st.dataframe(coverage, use_container_width=True)

# Status badge (existing)
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"

df_latest["status"] = df_latest.apply(status_badge, axis=1)

# Display snapshot table (existing)
cols = ["coin","date","close","wma_50","wma_200","position","previous_position","status","trend_img"]
st.subheader("Latest WMA snapshot per coin")
st.dataframe(
    df_latest[cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
        "trend_img": st.column_config.ImageColumn(
            "trend (last 30d + trendline)",
            width="small",
            help="Sparkline of last 30 closes with a linear-regression trendline overlay."
        ),
    },
)
