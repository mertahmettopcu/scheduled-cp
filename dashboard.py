import os
import re
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from supabase import create_client
from datetime import date

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

LEVERAGE = 10  # cross leverage

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
        st.error("Supabase credentials not found.")
        st.stop()
    return create_client(url, key)

supabase = supabase_client()

# -------------------- Data loaders --------------------
@st.cache_data(ttl=300)
def load_snapshot():
    snap = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(snap.data or [])
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_allocations():
    r = supabase.table("coin_allocations").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(ttl=300)
def load_spark_cache():
    r = supabase.table("coin_intraday_cache").select("*").execute()
    df = pd.DataFrame(r.data or [])
    return df

@st.cache_data(ttl=300)
def load_friend_decisions():
    try:
        r = supabase.table("friend_decisions").select("*").execute()
        df = pd.DataFrame(r.data or [])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        else:
            df = pd.DataFrame(columns=["coin","date","decision"])
        return df
    except Exception:
        return pd.DataFrame(columns=["coin","date","decision"])

def upsert_friend_decisions(rows: list[dict]) -> None:
    if not rows:
        return
    def _clean(r: dict) -> dict:
        out = {}
        out["coin"] = str(r.get("coin","")).upper().strip()
        d = r.get("date")
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

# -------------------- Sparkline helper --------------------
def build_spark_df():
    df = load_spark_cache()
    if df.empty:
        return pd.DataFrame(columns=["coin","spark_24h"])
    out = []
    for coin, sub in df.groupby("coin"):
        sub = sub.sort_values("hour")
        x = np.arange(len(sub))
        y = sub["close"].astype(float).to_numpy()
        if len(y) < 2:
            continue
        fig, ax = plt.subplots(figsize=(2.4,0.6), dpi=150)
        ax.plot(x, y, linewidth=1.0)
        # highlight noon idx if available
        if "noon_idx" in sub.columns:
            noon_idx = sub["noon_idx"].iloc[0]
            if 0 <= noon_idx < len(y):
                ax.scatter([noon_idx],[y[noon_idx]],color="red",s=20,zorder=3)
        ax.axis("off")
        ymin,ymax = np.nanmin(y),np.nanmax(y)
        pad = (ymax-ymin)*0.1 if ymax>ymin else 1
        ax.set_ylim(ymin-pad,ymax+pad)
        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append({"coin": coin, "spark_24h": f"data:image/png;base64,{b64}"})
    return pd.DataFrame(out)

# -------------------- Logic helpers --------------------
def ratio_from_decision(d: str) -> float:
    d = (d or "").lower()
    return 1.0 if "above" in d else 0.5 if "between" in d else 0.0

def op_from_prev_to_now(prev: str, now: str, amt: float) -> str:
    rp, rn = ratio_from_decision(prev), ratio_from_decision(now)
    delta = rn - rp
    if abs(delta) < 1e-12:
        return "No action"
    action = "Buy" if delta>0 else "Sell"
    qty = abs(delta)*max(amt,0.0)
    return f"{action} {qty:.6f}"

# -------------------- Main --------------------
st.title("📈 Crypto WMA Dashboard")

df_latest = load_snapshot()
df_alloc = load_allocations()
df_friend = load_friend_decisions()
spark_df = build_spark_df()

if df_latest.empty:
    st.info("No snapshot yet. Run pipeline first.")
    st.stop()

df_latest.columns = [c.lower() for c in df_latest.columns]
today_trt = date.today()

df_today_friend = df_friend[df_friend["date"]==today_trt][["coin","decision"]].rename(columns={"decision":"friend_today"})
df = (df_latest.merge(df_alloc,on="coin",how="left")
              .merge(spark_df,on="coin",how="left")
              .merge(df_today_friend,on="coin",how="left"))

df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["friend_decision"] = df["friend_today"].fillna(df["position"])

df["model_ratio"]  = df["position"].apply(ratio_from_decision)
df["friend_ratio"] = df["friend_decision"].apply(ratio_from_decision)
df["model_size"]   = df["allocation_amount"]*df["model_ratio"]
df["friend_size"]  = df["allocation_amount"]*df["friend_ratio"]
df["model_value_usd"]  = df["model_size"]*df["close"]*LEVERAGE
df["friend_value_usd"] = df["friend_size"]*df["close"]*LEVERAGE
df["model_op"] = df.apply(lambda r: op_from_prev_to_now(r.get("previous_position","Between"), r.get("position","Between"), float(r.get("allocation_amount",0.0))), axis=1)

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
    "allocation_amount": st.column_config.NumberColumn(format="%.6f"),
    "model_size": st.column_config.NumberColumn(format="%.6f"),
    "friend_size": st.column_config.NumberColumn(format="%.6f"),
    "model_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_decision": st.column_config.SelectboxColumn(
        "friend_decision", options=["Above both","Between","Below both"]
    ),
    "spark_24h": st.column_config.ImageColumn("spark_24h", width="small"),
}

disabled = [c for c in view_cols if c!="friend_decision"]

edited = st.data_editor(
    df[view_cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config=colcfg,
    disabled=disabled,
    key="snapshot_editor",
)

if st.button("💾 Save friend decisions"):
    before = df.set_index("coin")["friend_decision"]
    after = pd.DataFrame(edited).set_index("coin")["friend_decision"]
    changed = after[after!=before].dropna()
    payload = [{"coin":c,"date":today_trt,"decision":v} for c,v in changed.items()]
    upsert_friend_decisions(payload)
    st.success(f"Saved {len(payload)} change(s). Please rerun to refresh values.")

# ---- Portfolio chart ----
st.subheader("Portfolio value comparison (Model vs Friend)")
df_curve = df.groupby("coin")[["model_value_usd","friend_value_usd"]].sum().sum()
model_total = df["model_value_usd"].sum()
friend_total = df["friend_value_usd"].sum()
st.metric("Model total (USD)", f"{model_total:,.2f}")
st.metric("Friend total (USD)", f"{friend_total:,.2f}")
