import os
import re
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

SPARK_DAYS = 30  # how many days to show in the small chart

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

# -------------------- Data loader --------------------
@st.cache_data(ttl=300)
def load_data():
    # Latest snapshot (one row per coin)
    snap = supabase.table("coin_wma_latest").select("*").execute()
    df_latest = pd.DataFrame(snap.data or [])

    # -------- Paged fetch for history (to bypass the 1000-row default limit) --------
    page_size = 2000  # safe cushion; can adjust up/down
    offset = 0
    hist_rows = []

    # You get the most predictable results if you ORDER before you RANGE
    # (PostgREST applies range after ordering).
    # We request only the columns we need.
    while True:
        resp = (
            supabase
            .table("coin_wma")
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
        # defensive stop (in case someone removes ordering), but practically not needed
        if len(batch) < page_size:
            break

    df_hist = pd.DataFrame(hist_rows)
    return df_latest, df_hist

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

# -------------------- Tiny chart (sparkline + trendline) --------------------
def _spark_with_trend_png(y: np.ndarray) -> str | None:
    """
    Given a 1D array of closes, render a small chart:
    - series (thin line)
    - linear regression trendline (dashed, slightly thicker)
    Returns a data URI (base64 PNG) suitable for ImageColumn, or None if not enough data.
    """
    if y is None or len(y) < 2 or np.isnan(y).all():
        return None

    # Clean NaNs (drop; keep relative index spacing)
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    x_masked = x[mask]
    y_masked = y[mask]
    if len(y_masked) < 5:
        return None

    # Fit simple linear regression y = a*x + b
    a, b = np.polyfit(x_masked, y_masked, 1)
    y_fit = a * x + b

    # Plot tiny, minimal chart
    fig, ax = plt.subplots(figsize=(2.4, 0.6), dpi=150)
    # series
    ax.plot(x, y, linewidth=1.0)
    # trendline
    ax.plot(x, y_fit, linestyle="--", linewidth=1.2)

    # remove decorations
    ax.axis("off")
    # a bit of headroom
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
    """
    For each coin, take its last `days` closes and build a PNG data URI
    with sparkline + trendline. Returns a DataFrame with columns [coin, trend_img].
    """
    if df_hist.empty:
        return pd.DataFrame(columns=["coin", "trend_img"])

    df = df_hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Take last N per coin
    tailn = (
        df.sort_values(["coin", "date"])
          .groupby("coin", as_index=False, group_keys=False)
          .tail(days)
    )

    images = []
    for coin, sub in tailn.groupby("coin"):
        y = sub["close"].astype(float).to_numpy()
        img = _spark_with_trend_png(y)
        images.append({"coin": coin, "trend_img": img})
    return pd.DataFrame(images)

# Build trend images and merge onto snapshot
trend_imgs = build_trend_images(df_hist, SPARK_DAYS)
df_latest = df_latest.merge(trend_imgs, on="coin", how="left")

# status (Change if position != previous_position)
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"

df_latest["status"] = df_latest.apply(status_badge, axis=1)

# -------------------- Display --------------------
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
