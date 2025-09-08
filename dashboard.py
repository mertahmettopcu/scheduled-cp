 #!/usr/bin/env python3
 # dashboard.py - Streamlit app for coin_wma table
 import streamlit as st
 import pandas as pd
 from supabase import create_client
 SUPABASE_URL = st.secrets.get("SUPABASE_URL") or "https://YOUR-PROJECT-URL.supabase.co"
 SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or "YOUR-ANON-KEY"
 supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
 st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")
 st.title("Crypto WMA Dashboard (Noon Turkey-time closes)")
 st.markdown("Pick a coin to see history, or browse latest snapshot.")
 @st.cache_data(ttl=300)
 def load_data():
    data = supabase.table("coin_wma").select("*").execute().data
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df
 df = load_data()
 if df.empty:
    st.warning("No data in Supabase yet. Run the pipeline at least once.")
 else:
    coins = sorted(df["coin"].unique())
    selected = st.sidebar.selectbox("Coin", coins)
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    dr = st.sidebar.date_input("Date range", [date_min, date_max])
    filtered = df[(df["coin"]==selected) & (df["date"]>=pd.to_datetime(dr[0])) & (df["date"]<=pd.to_datetime(dr[1]))].sort_values("date")
    st.subheader("Latest snapshot for all coins")
    latest = df.sort_values("date").groupby("coin").tail(1)
    st.dataframe(latest[["coin","date","close","WMA_50","WMA_200","position","previous_position"]])
    st.subheader(f"{selected} history")
    st.line_chart(filtered.set_index("date")[["close","wma_50","wma_200"]].rename(columns={"wma_50":"WMA_50","wma_200":"WMA_200"}))
    st.subheader("Quick previews (last 30 days)")
    last30 = df[df["date"] > df["date"].max() - pd.Timedelta(days=30)]
    cols = st.columns(5)
    for i, coin in enumerate(sorted(last30["coin"].unique())[:15]):
        sub = last30[last30["coin"]==coin]
        with cols[i % 5]:
            st.caption(coin)
            if not sub.empty:
                st.line_chart(sub.set_index("date")["close"])