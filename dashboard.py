# dashboard.py
import re
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

# -------------------- Supabase client (from Streamlit secrets) --------------------
@st.cache_resource
def supabase_client():
    url = st.secrets["supabase"]["url"].strip()
    key = st.secrets["supabase"]["key"].strip()
    # remove any sneaky non-breaking spaces
    url = re.sub(r"\u00A0", " ", url).strip()
    key = re.sub(r"\u00A0", " ", key).strip()
    return create_client(url, key)

supabase = supabase_client()

# -------------------- Data loader --------------------
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    resp = supabase.table("coin_wma").select("*").execute()
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Map any legacy names to the new snake_case (defensive)
    rename_map = {
        "wma_50": "wma_50",
        "wma50": "wma_50",
        "wma_200": "wma_200",
        "wma200": "wma_200",
        "position": "position",
        "previous_position": "previous_position",
        "close": "close",
        "coin": "coin",
        "date": "date",
    }
    df = df.rename(columns=rename_map)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Basic type coercion (optional)
    for col in ["close", "wma_50", "wma_200"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

st.title("📈 Crypto WMA Dashboard")

if df.empty:
    st.info("No data available yet. Populate the table and refresh.")
else:
    needed = ["coin", "date", "close", "wma_50", "wma_200", "position", "previous_position"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        st.error(f"Missing columns in data: {missing}\nAvailable columns: {list(df.columns)}")
    else:
        # Latest row per coin
        latest = (
            df.sort_values(["coin", "date"])
              .groupby("coin", as_index=False)
              .tail(1)
              .sort_values("coin")
              .reset_index(drop=True)
        )

        st.subheader("Latest WMA snapshot per coin")
        st.dataframe(latest[needed], use_container_width=True)

        # Recent 30 days (optional)
        if pd.notna(df["date"]).any():
            st.subheader("Recent rows (last 30 days)")
            recent_cut = df["date"].max() - pd.Timedelta(days=30)
            recent = df[df["date"] >= recent_cut].sort_values(["coin", "date"])
            st.dataframe(recent[needed], use_container_width=True)
