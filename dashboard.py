import os
import re
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

SPARK_DAYS = 30  # days for sparkline

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

# Build per-coin trend from the last SPARK_DAYS rows (no global cutoff)
trend = None
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

    # pack into lists for st.column_config.LineChartColumn
    trend = (
        tailn.groupby("coin")["close"]
             .apply(lambda s: [float(x) for x in s.tolist()])
             .rename("trend")
             .reset_index()
    )

    # merge onto latest snapshot
    df_latest = df_latest.merge(trend, on="coin", how="left")



# status (Change if position != previous_position)
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"

df_latest["status"] = df_latest.apply(status_badge, axis=1)

# Display
cols = ["coin","date","close","wma_50","wma_200","position","previous_position","status","trend"]
st.subheader("Latest WMA snapshot per coin")
st.dataframe(
    df_latest[cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
        # slimmer sparkline
        "trend": st.column_config.LineChartColumn("trend (last 30d)", width="small"),
    },
)
