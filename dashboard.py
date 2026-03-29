import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto Futures Dashboard", layout="wide")

# =========================================================
# Config / Supabase
# =========================================================
PAIR_OPTIONS = ["BTCUSDT", "PAXGUSDT"]
NOON_TZ = "Europe/Istanbul"


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

    if url:
        url = re.sub(r"\u00A0", " ", url).strip()
    if key:
        key = re.sub(r"\u00A0", " ", key).strip()
    return url, key


@st.cache_resource
def supabase_client():
    url, key = _read_supabase_creds()
    if not url or not key:
        st.error("Supabase credentials not found. Add in Streamlit secrets or env vars.")
        st.stop()
    return create_client(url, key)


supabase = supabase_client()


# =========================================================
# Loaders
# =========================================================
@st.cache_data(ttl=300)
def load_candles(pair: str, timeframe: str) -> pd.DataFrame:
    r = (
        supabase.table("futures_candles")
        .select("*")
        .eq("pair", pair)
        .eq("timeframe", timeframe)
        .order("open_time", desc=False)
        .execute()
    )
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)
    num_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("open_time").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_signal_snapshots(pair: str) -> pd.DataFrame:
    r = (
        supabase.table("futures_signal_snapshots")
        .select("*")
        .eq("pair", pair)
        .order("timeframe", desc=False)
        .execute()
    )
    df = pd.DataFrame(r.data or [])
    if not df.empty and "last_open_time" in df.columns:
        df["last_open_time"] = pd.to_datetime(df["last_open_time"], errors="coerce", utc=True)
    return df


# =========================================================
# Indicator helpers
# =========================================================
def add_ema_rsi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for length in (4, 16, 65, 120):
        out[f"ema{length}"] = out["close"].ewm(span=length, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    def _rsi(period: int) -> pd.Series:
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    out["rsi14"] = _rsi(14)
    out["rsi52"] = _rsi(52)
    out["ema4_slope"] = out["ema4"].diff()
    out["ema16_slope"] = out["ema16"].diff()

    prev_ema4 = out["ema4"].shift(1)
    prev_ema16 = out["ema16"].shift(1)
    out["long_signal"] = (
        (out["ema4"] > out["ema16"]) &
        (prev_ema4 <= prev_ema16) &
        (out["rsi14"] > out["rsi52"])
    )
    out["short_signal"] = (
        (out["ema4"] < out["ema16"]) &
        (prev_ema4 >= prev_ema16) &
        (out["rsi14"] < out["rsi52"])
    )
    return out



def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high_9 = out["high"].rolling(9).max()
    low_9 = out["low"].rolling(9).min()
    out["tenkan"] = (high_9 + low_9) / 2

    high_26 = out["high"].rolling(26).max()
    low_26 = out["low"].rolling(26).min()
    out["kijun"] = (high_26 + low_26) / 2

    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(26)

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()
    out["senkou_b"] = ((high_52 + low_52) / 2).shift(26)

    out["chikou"] = out["close"].shift(-26)
    return out


# =========================================================
# Chart builders
# =========================================================
def make_price_ema_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    for length in (4, 16, 65, 120):
        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df[f"ema{length}"],
                mode="lines",
                name=f"EMA{length}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        hovermode="x unified",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
    )
    return fig



def make_ichimoku_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    for col, label in [
        ("tenkan", "Tenkan"),
        ("kijun", "Kijun"),
        ("senkou_a", "Senkou A"),
        ("senkou_b", "Senkou B"),
        ("chikou", "Chikou"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df[col],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        hovermode="x unified",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
    )
    return fig


# =========================================================
# UI
# =========================================================
st.title("Crypto Futures Strategy Dashboard")
st.caption("Perpetual futures candles come from Supabase cache populated by your GitHub pipeline.")

selected_pair = st.selectbox("Crypto seç", PAIR_OPTIONS, index=0)

snapshots = load_signal_snapshots(selected_pair)

hourly = load_candles(selected_pair, "1h")
m15 = load_candles(selected_pair, "15m")
daily = load_candles(selected_pair, "1d")

if hourly.empty or m15.empty or daily.empty:
    st.warning("Bu coin için gerekli candle verileri Supabase'te yok. Önce pipeline'ı çalıştır.")
    st.stop()

hourly = add_ema_rsi(hourly)
m15 = add_ema_rsi(m15)
daily = add_ichimoku(daily)

hourly_chart = hourly.tail(48).copy()   # last 2 days of 1h candles
a15_chart = m15.tail(96).copy()         # last 1 day of 15m candles
daily_chart = daily.tail(220).copy()

# Latest metrics
h_last = hourly.iloc[-1]
m15_last = m15.iloc[-1]
d_last = daily.iloc[-1]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1H latest**")
    st.write({
        "close": round(float(h_last["close"]), 4),
        "ema4": round(float(h_last["ema4"]), 4),
        "ema16": round(float(h_last["ema16"]), 4),
        "ema65": round(float(h_last["ema65"]), 4),
        "ema120": round(float(h_last["ema120"]), 4),
        "rsi14": None if pd.isna(h_last["rsi14"]) else round(float(h_last["rsi14"]), 2),
        "rsi52": None if pd.isna(h_last["rsi52"]) else round(float(h_last["rsi52"]), 2),
        "signal": "LONG" if bool(h_last["long_signal"]) else "SHORT" if bool(h_last["short_signal"]) else "NONE",
    })
with c2:
    st.markdown("**15M latest**")
    st.write({
        "close": round(float(m15_last["close"]), 4),
        "ema4": round(float(m15_last["ema4"]), 4),
        "ema16": round(float(m15_last["ema16"]), 4),
        "ema65": round(float(m15_last["ema65"]), 4),
        "ema120": round(float(m15_last["ema120"]), 4),
        "rsi14": None if pd.isna(m15_last["rsi14"]) else round(float(m15_last["rsi14"]), 2),
        "rsi52": None if pd.isna(m15_last["rsi52"]) else round(float(m15_last["rsi52"]), 2),
        "signal": "LONG" if bool(m15_last["long_signal"]) else "SHORT" if bool(m15_last["short_signal"]) else "NONE",
    })
with c3:
    st.markdown("**1D Ichimoku latest**")
    st.write({
        "close": round(float(d_last["close"]), 4),
        "tenkan": None if pd.isna(d_last["tenkan"]) else round(float(d_last["tenkan"]), 4),
        "kijun": None if pd.isna(d_last["kijun"]) else round(float(d_last["kijun"]), 4),
        "senkou_a": None if pd.isna(d_last["senkou_a"]) else round(float(d_last["senkou_a"]), 4),
        "senkou_b": None if pd.isna(d_last["senkou_b"]) else round(float(d_last["senkou_b"]), 4),
        "chikou": None if pd.isna(d_last["chikou"]) else round(float(d_last["chikou"]), 4),
    })

if not snapshots.empty:
    st.subheader("Latest pipeline snapshot")
    snap_view = snapshots.copy()
    st.dataframe(snap_view.sort_values("timeframe"), use_container_width=True)

st.subheader(f"{selected_pair} — 1H (son 2 gün)")
st.plotly_chart(
    make_price_ema_chart(hourly_chart, f"{selected_pair} 1H — Price + EMA4/16/65/120"),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 15M (son 1 gün)")
st.plotly_chart(
    make_price_ema_chart(a15_chart, f"{selected_pair} 15M — Price + EMA4/16/65/120"),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 1D Ichimoku")
st.plotly_chart(
    make_ichimoku_chart(daily_chart, f"{selected_pair} 1D — Ichimoku"),
    use_container_width=True,
)
