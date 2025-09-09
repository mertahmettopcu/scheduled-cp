import os
import re
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

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

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    resp = supabase.table("coin_wma").select("*").execute()
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    # parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # coerce numerics
    for col in ["close", "wma_50", "wma_200"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

st.title("📈 Crypto WMA Dashboard")

df = load_data()
if df.empty:
    st.info("No data yet. Try running the pipeline, then refresh.")
else:
    needed = ["coin", "date", "close", "wma_50", "wma_200", "position", "previous_position"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
    else:
        latest = (
            df.sort_values(["coin", "date"])
              .groupby("coin", as_index=False)
              .tail(1)
              .sort_values("coin")
              .reset_index(drop=True)
        )
        st.subheader("Latest WMA snapshot per coin")
        st.dataframe(latest[needed], use_container_width=True)

        st.subheader("Recent rows (last 30 days)")
        recent_cut = df["date"].max() - pd.Timedelta(days=30)
        recent = df[df["date"] >= recent_cut].sort_values(["coin", "date"])
        st.dataframe(recent[needed], use_container_width=True)
