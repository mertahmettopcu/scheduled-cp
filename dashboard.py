import os
import re
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

SPARK_DAYS = 30  # how many days to show in the sparkline

# -------------------- Supabase creds --------------------
def _read_supabase_creds():
    url = None
    key = None
    # Try Streamlit secrets first
    try:
        if "supabase" in st.secrets:
            url = st.secrets["supabase"].get("url")
            key = st.secrets["supabase"].get("key")
    except Exception:
        pass
    # Fallback to environment variables
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_KEY")
    # Clean any sneaky non-breaking spaces
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
def load_data() -> pd.DataFrame:
    PAGE_SIZE = 5000  # generous; tune as you like
    rows = []
    start = 0

    while True:
        end = start + PAGE_SIZE - 1
        resp = (
            supabase
            .table("coin_wma")
            .select("*")
            .range(start, end)      # <-- fetch a page
            .order("date")          # keep deterministic across pages
            .execute()
        )
        batch = getattr(resp, "data", None) or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize columns
    df.columns = [c.lower() for c in df.columns]

    # parse/coerce
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["close", "wma_50", "wma_200"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------- UI --------------------
st.title("📈 Crypto WMA Dashboard")

df = load_data()
if df.empty:
    st.info("No data yet. Try running the pipeline, then refresh.")
    st.stop()

needed = ["coin", "date", "close", "wma_50", "wma_200", "position", "previous_position"]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

# Header stats
last_dt = pd.to_datetime(df["date"]).max()
unique_coins = df["coin"].nunique()
c1, c2 = st.columns(2)
c1.metric("Coins shown", f"{unique_coins}")
c2.metric("Last updated (UTC)", last_dt.strftime("%Y-%m-%d") if pd.notnull(last_dt) else "—")

# Latest row per coin
latest = (
    df.sort_values(["coin", "date"])
      .groupby("coin", as_index=False)
      .tail(1)
      .sort_values("coin")
      .reset_index(drop=True)
)

# Build sparkline data (last SPARK_DAYS closes per coin)
cutoff = df["date"].max() - pd.Timedelta(days=SPARK_DAYS)
recent = (
    df[df["date"] >= cutoff]
    .sort_values(["coin", "date"])[["coin", "date", "close"]]
)

series = (
    recent.groupby("coin")["close"]
          .apply(lambda s: [float(x) for x in s.tolist()])
          .rename("trend")
          .reset_index()
)

latest = latest.merge(series, on="coin", how="left")

# Add status column to highlight changes in position
latest["change"] = latest.apply(
    lambda r: "🔔 Change" if r["position"] != r["previous_position"] else "✓ Stable",
    axis=1
)

# Show table with sparkline column
show_cols = ["coin", "date", "close", "wma_50", "wma_200", "position", "previous_position", "change", "trend"]

st.subheader("Latest WMA snapshot per coin")
st.dataframe(
    latest[show_cols],
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
        "change": st.column_config.TextColumn("status"),
        "trend": st.column_config.LineChartColumn(
            "trend (last 30d)",
            width="small",
            y_min=None,
            y_max=None,
        ),
    },
)
