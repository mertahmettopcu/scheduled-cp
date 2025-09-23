import os
import re
import io
import json
import base64
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

# Constants
LEVERAGE = 10.0            # 10x cross
NOON_TZ = "Europe/Istanbul"

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
        st.error("Supabase credentials not found.\nAdd in Secrets or env vars.")
        st.stop()
    return create_client(url, key)

supabase = supabase_client()

# -------------------- Data loaders --------------------
@st.cache_data(ttl=300)
def load_snapshot():
    r = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ("close","wma_50","wma_200"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_allocations():
    r = supabase.table("coin_allocations").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df["coin"] = df["coin"].astype(str)
        df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(ttl=300)
def load_intraday_cache():
    # ✅ correct table name + schema from the pipeline
    r = supabase.table("coin_hourly_24h_cache").select("*").execute()
    df = pd.DataFrame(r.data or [])
    return df

@st.cache_data(ttl=300)
def load_friend_decisions():
    r = supabase.table("friend_decisions").select("*").execute()
    data = r.data or []
    if not data:
        # ensure columns exist even when empty
        return pd.DataFrame(columns=["coin","date","decision"])
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df

# -------------------- Friend decisions upsert (JSON-safe) --------------------
def upsert_friend_decisions(rows: list[dict]) -> None:
    if not rows:
        return

    def _clean(r: dict) -> dict:
        out = {}
        out["coin"] = str(r.get("coin","")).upper().strip()
        d = r.get("date")
        # normalize date -> YYYY-MM-DD
        if isinstance(d, pd.Timestamp):
            d = d.date()
        if hasattr(d, "isoformat"):
            out["date"] = d.isoformat()
        else:
            out["date"] = str(d)
        out["decision"] = str(r.get("decision","")).strip()
        return out

    for r in rows:
        supabase.table("friend_decisions").upsert(
            _clean(r), on_conflict="coin,date", returning="minimal"
        ).execute()

# -------------------- Sparkline renderer (hourly cache JSON) --------------------
def render_spark_24h(series, noon_index, w50, w200) -> str | None:
    """
    series: list of {"t": iso8601 UTC string, "c": float}
    noon_index: int index into the series (0..23)
    Returns base64 data URI or None.
    """
    if not series or len(series) < 2:
        return None
    # parse closes
    closes = np.array([float(p.get("c", np.nan)) for p in series], dtype=float)
    x = np.arange(len(closes))
    mask = ~np.isnan(closes)
    if mask.sum() < 2:
        return None

    # trendline
    a, b = np.polyfit(x[mask], closes[mask], 1)
    y_fit = a * x + b

    fig, ax = plt.subplots(figsize=(2.6, 0.6), dpi=150)
    # price line
    ax.plot(x, closes, linewidth=1.0)
    # trendline
    ax.plot(x, y_fit, linestyle="--", linewidth=1.0)
    # noon marker
    if noon_index is not None and 0 <= int(noon_index) < len(closes) and np.isfinite(closes[int(noon_index)]):
        ax.scatter([int(noon_index)], [closes[int(noon_index)]], s=14, zorder=5)
    # horizontal WMA refs (daily)
    if isinstance(w50, (int, float)) and np.isfinite(w50):
        ax.axhline(w50, color="green", linestyle=":", linewidth=1.0, label="WMA50")
    if isinstance(w200, (int, float)) and np.isfinite(w200):
        ax.axhline(w200, color="orange", linestyle="--", linewidth=1.0, label="WMA200")


    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    ymin, ymax = np.nanmin(closes[mask]), np.nanmax(closes[mask])
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def build_spark_df(df_latest: pd.DataFrame, df_cache: pd.DataFrame) -> pd.DataFrame:
    """Merge cached JSON series onto coins and render base64 PNGs."""
    if df_cache.empty:
        return pd.DataFrame(columns=["coin","spark_24h"])
    imgs = []
    cache_map = {r["coin"]: r for r in df_cache.to_dict(orient="records")}
    for _, r in df_latest.iterrows():
        coin = r["coin"]
        entry = cache_map.get(coin)
        if not entry:
            imgs.append({"coin": coin, "spark_24h": None})
            continue
        series = entry.get("series", [])
        if isinstance(series, str):
            try:
                series = json.loads(series)
            except Exception:
                series = []
        noon_idx = entry.get("noon_index")
        img = render_spark_24h(series, noon_idx, r.get("wma_50"), r.get("wma_200"))
        imgs.append({"coin": coin, "spark_24h": img})
    return pd.DataFrame(imgs)

# -------------------- Logic helpers --------------------
def ratio_from_decision(d: str) -> float:
    d = (d or "").strip().lower()
    return 1.0 if d == "above both" else 0.5 if d == "between" else 0.0

def op_from_prev_to_now(prev: str, now: str, amt: float) -> str:
    rp, rn = ratio_from_decision(prev), ratio_from_decision(now)
    delta = rn - rp
    if abs(delta) < 1e-12:
        return "No action"
    action = "Buy" if delta > 0 else "Sell"
    qty = abs(delta) * max(amt, 0.0)
    return f"{action} {qty:.6f}"

# -------------------- Page --------------------
st.title("📈 Crypto WMA Dashboard")

df_latest = load_snapshot()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh.")
    st.stop()

df_alloc = load_allocations()
df_cache = load_intraday_cache()
df_friend = load_friend_decisions()

# Header
last_dt = df_latest["date"].max()
coins_shown = df_latest["coin"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Coins shown", f"{coins_shown}")
c2.metric("Last updated (UTC)", last_dt.strftime("%Y-%m-%d") if pd.notnull(last_dt) else "—")
c3.metric("Leverage (x)", f"{LEVERAGE:g}")

# Inline table with friend editor
today_trt = datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=NOON_TZ).tz).date()

# Spark images
spark_df = build_spark_df(df_latest, df_cache)

# friend (today) merge
df_today_friend = df_friend[df_friend["date"] == today_trt][["coin","decision"]].rename(columns={"decision":"friend_today"})

# assemble view df
df = (df_latest
      .merge(df_alloc, on="coin", how="left")
      .merge(df_today_friend, on="coin", how="left")
      .merge(spark_df, on="coin", how="left"))

df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["friend_decision"] = df["friend_today"].fillna(df["position"])

# sizes & USD values (today)
df["model_ratio"]  = df["position"].apply(ratio_from_decision)
df["friend_ratio"] = df["friend_decision"].apply(ratio_from_decision)

df["model_size"]   = df["allocation_amount"] * df["model_ratio"]
df["friend_size"]  = df["allocation_amount"] * df["friend_ratio"]

df["model_value_usd"]  = df["model_size"]  * df["close"] * LEVERAGE
df["friend_value_usd"] = df["friend_size"] * df["close"] * LEVERAGE

# model operation delta from previous_position
df["model_op"] = df.apply(
    lambda r: op_from_prev_to_now(
        r.get("previous_position","Between"),
        r.get("position","Between"),
        float(r.get("allocation_amount",0.0))
    ),
    axis=1
)

# Data editor
st.subheader("Latest snapshot (inline friend decisions)")
view_cols = [
    "coin","date","close","wma_50","wma_200","position","previous_position",
    "model_op","allocation_amount","model_size","friend_decision","friend_size",
    "model_value_usd","friend_value_usd","spark_24h"
]

colcfg = {
    "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
    "close": st.column_config.NumberColumn(format="%.6f"),
    "wma_50": st.column_config.NumberColumn(format="%.6f"),
    "wma_200": st.column_config.NumberColumn(format="%.6f"),
    "allocation_amount": st.column_config.NumberColumn(format="%.6f",
        help="Full coin amount when Above; Between=half; Below=0"),
    "model_size": st.column_config.NumberColumn(format="%.6f"),
    "friend_size": st.column_config.NumberColumn(format="%.6f"),
    "model_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_decision": st.column_config.SelectboxColumn(
        "friend_decision",
        options=["Above both","Between","Below both"],
        help=f"Today's friend decision ({today_trt.isoformat()})"
    ),
    "spark_24h": st.column_config.ImageColumn("spark_24h", width="small",
        help="Hourly last 24h + trendline + noon marker + horizontal daily WMAs"),
}

disabled = [c for c in view_cols if c != "friend_decision"]

edited = st.data_editor(
    df[view_cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config=colcfg,
    disabled=disabled,
    key="snapshot_editor",
)

if st.button("💾 Save friend decisions (today)"):
    before = df.set_index("coin")["friend_decision"]
    after = pd.DataFrame(edited).set_index("coin")["friend_decision"]
    
    # align indexes so compare works
    before, after = before.align(after, join="outer")
    
    changed = after[after != before].dropna()

    payload = [{"coin": c, "date": today_trt, "decision": v} for c, v in changed.items()]
    upsert_friend_decisions(payload)
    st.success(f"Saved {len(payload)} change(s). Please rerun to refresh values.")

# -------------------- Portfolio snapshot metrics (today) --------------------
st.subheader("Portfolio snapshot — current notional (USD, 10×)")
model_total = float(df["model_value_usd"].sum())
friend_total = float(df["friend_value_usd"].sum())
cA, cB, cC = st.columns(3)
cA.metric("Model total (USD)", f"{model_total:,.2f}")
cB.metric("Friend total (USD)", f"{friend_total:,.2f}")
cC.metric("Difference (USD)", f"{(model_total - friend_total):,.2f}")
