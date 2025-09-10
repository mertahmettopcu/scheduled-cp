import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

SPARK_DAYS = 30  # days for sparkline
TREND_STYLE = "regression"  # "regression" or "wma"
WMA_WINDOW = 10             # only used when TREND_STYLE == "wma"

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
    # Canonical latest snapshot (one row per coin)
    snap = supabase.table("coin_wma_latest").select("*").execute()
    df_latest = pd.DataFrame(snap.data or [])
    # History for sparkline
    hist = supabase.table("coin_wma").select("coin,date,close").execute()
    df_hist = pd.DataFrame(hist.data or [])
    return df_latest, df_hist

# -------------------- Tiny chart helpers --------------------
def _wma(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    weights = np.arange(1, window + 1, dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        seg = values[i - window + 1 : i + 1]
        if np.isnan(seg).any():
            continue
        out[i] = (seg * weights).sum() / weights.sum()
    return out

def make_spark_with_trend(y: np.ndarray) -> bytes | None:
    """Return a tiny PNG (bytes) of close line + trend line."""
    y = y.astype(float)
    y = y[~np.isnan(y)]
    if y.size < 2:
        return None

    x = np.arange(len(y))

    # figure size tuned so the image looks like the built-in spark
    fig, ax = plt.subplots(figsize=(2.6, 0.7), dpi=150)
    # price line
    ax.plot(x, y, color="red", linewidth=1)

    # overlay trend
    if TREND_STYLE == "regression":
        # simple linear regression trend
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
    else:  # TREND_STYLE == "wma"
        trend = _wma(y, min(WMA_WINDOW, len(y)))

    # only plot trend where it exists
    if np.isfinite(trend).any():
        ax.plot(x[np.isfinite(trend)], trend[np.isfinite(trend)],
                color="black", linewidth=1, linestyle="--")

    # clean look
    ax.axis("off")
    # ensure a tight layout to remove margins
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# -------------------- UI --------------------
st.title("📈 Crypto WMA Dashboard")

df_latest, df_hist = load_data()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh.")
    st.stop()

# Normalize latest
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

# Build per-coin spark images from the last SPARK_DAYS rows
spark_df = None
if not df_hist.empty:
    df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
    df_hist["close"] = pd.to_numeric(df_hist["close"], errors="coerce")

    # sort and take last N per coin
    tailn = (
        df_hist.sort_values(["coin", "date"])
               .groupby("coin", as_index=False, group_keys=False)
               .tail(SPARK_DAYS)
               .dropna(subset=["close"])
    )

    # turn each coin's last N closes into a tiny PNG with trend overlay
    imgs = (
        tailn.groupby("coin")["close"]
             .apply(lambda s: make_spark_with_trend(s.to_numpy()))
             .rename("spark")
             .reset_index()
    )

    spark_df = imgs
    # merge onto latest snapshot
    df_latest = df_latest.merge(spark_df, on="coin", how="left")

# status (Change if position != previous_position)
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"

df_latest["status"] = df_latest.apply(status_badge, axis=1)

# Display
cols = ["coin","date","close","wma_50","wma_200","position","previous_position","status","spark"]
st.subheader("Latest WMA snapshot per coin")
st.dataframe(
    df_latest[cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
        "spark": st.column_config.ImageColumn("trend (last 30d)", width="small"),
    },
)
