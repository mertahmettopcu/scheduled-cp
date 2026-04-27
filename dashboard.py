import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Crypto Futures Dashboard", layout="wide")

st.markdown("""
<style>
footer {visibility: hidden;}
header {visibility: hidden;}

/* Streamlit sağ alt / profil / deploy kontrol alanlarını gizlemeyi dener */
[data-testid="stStatusWidget"] {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

[data-testid="stMainMenu"] {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

def load_allowed_emails() -> set[str]:
    users_csv_url = st.secrets["access"]["users_csv_url"]
    df = pd.read_csv(users_csv_url)

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "email" not in df.columns or "active" not in df.columns:
        st.error("Users sheet must contain 'email' and 'active' columns.")
        st.stop()

    active_mask = (
        df["active"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )

    emails = (
        df.loc[active_mask, "email"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return set(emails.tolist())

params = st.query_params

if params.get("ping") == st.secrets["access"]["keepalive_token"]:
    st.write("ok")
    st.stop()
    
if not st.user.is_logged_in:
    st.title("Giriş gerekli")
    if st.button("Google ile giriş yap"):
        st.login()
    st.stop()

user_email = (st.user.get("email") or "").lower().strip()
allowed_emails = load_allowed_emails()

#st.sidebar.button("Çıkış yap", on_click=st.logout)
if st.button("Çıkış yap"):
    st.logout()

if user_email not in allowed_emails:
    st.error("Erişim izniniz yok")
    st.stop()

# =========================================================
# Config / Supabase
# =========================================================
PAIR_OPTIONS = ["BTCUSDT", "PAXGUSDT"]
DISPLAY_TZ = "Europe/Istanbul"


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
# Helpers
# =========================================================
def _to_display_time(ts):
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(DISPLAY_TZ)


def _fmt_display_time(ts):
    ts = _to_display_time(ts)
    if ts is None:
        return None
    return ts.strftime("%Y-%m-%d %H:%M")


def _safe_round(value, digits=4):
    if pd.isna(value):
        return None
    return round(float(value), digits)


# =========================================================
# Loaders
# =========================================================
@st.cache_data(ttl=60)
def load_candles(pair: str, timeframe: str) -> pd.DataFrame:
    r = (
        supabase.table("futures_candles")
        .select("*")
        .eq("pair", pair)
        .eq("timeframe", timeframe)
        .order("open_time", desc=True)
        .limit(500)
        .execute()
    )
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)

    num_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=60)
def load_signal_snapshots(pair: str) -> pd.DataFrame:
    r = (
        supabase.table("futures_signal_snapshots")
        .select("*")
        .eq("pair", pair)
        .execute()
    )
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    if "last_open_time" in df.columns:
        df["last_open_time"] = pd.to_datetime(df["last_open_time"], errors="coerce", utc=True)
    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)

    return df.sort_values("timeframe").reset_index(drop=True)


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

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()

    # unshifted future cloud base values
    out["senkou_a_base"] = (out["tenkan"] + out["kijun"]) / 2
    out["senkou_b_base"] = (high_52 + low_52) / 2

    # standard plotted / aligned values
    out["senkou_a"] = out["senkou_a_base"].shift(26)
    out["senkou_b"] = out["senkou_b_base"].shift(26)

    out["chikou"] = out["close"].shift(-26)
    return out


# =========================================================
# Chart builders
# =========================================================
def make_price_ema_chart(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df.copy()
    plot_df["display_time"] = plot_df["open_time"].dt.tz_convert(DISPLAY_TZ)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=plot_df["display_time"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        )
    )

    for length in (4, 16, 65, 120):
        fig.add_trace(
            go.Scatter(
                x=plot_df["display_time"],
                y=plot_df[f"ema{length}"],
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
    plot_df = df.copy()
    plot_df["display_time"] = plot_df["open_time"].dt.tz_convert(DISPLAY_TZ)

    # Future 26-day extension using unshifted values
    future_src = plot_df.tail(26).copy()
    future_src["display_time"] = future_src["display_time"] + pd.Timedelta(days=26)
    future_src["senkou_a"] = future_src["senkou_a_base"]
    future_src["senkou_b"] = future_src["senkou_b_base"]

    # Visible cloud + future cloud together
    cloud_df = pd.concat(
        [
            plot_df[["display_time", "senkou_a", "senkou_b"]],
            future_src[["display_time", "senkou_a", "senkou_b"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["display_time"]).sort_values("display_time").reset_index(drop=True)

    fig = go.Figure()

    # Price
    fig.add_trace(
        go.Candlestick(
            x=plot_df["display_time"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        )
    )

    # Tenkan / Kijun / Chikou
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["tenkan"],
            mode="lines",
            name="Tenkan",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["kijun"],
            mode="lines",
            name="Kijun",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["chikou"],
            mode="lines",
            name="Chikou",
        )
    )

    # Senkou outlines
    fig.add_trace(
        go.Scatter(
            x=cloud_df["display_time"],
            y=cloud_df["senkou_a"],
            mode="lines",
            name="Senkou A",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cloud_df["display_time"],
            y=cloud_df["senkou_b"],
            mode="lines",
            name="Senkou B",
            line=dict(width=2),
        )
    )

    # -------- segmented cloud fill helper --------
    def add_cloud_segments(mask, fillcolor, name):
        segment_id = (mask != mask.shift()).cumsum()

        first_segment = True
        for _, seg in cloud_df[mask].groupby(segment_id[mask]):
            if len(seg) < 2:
                continue

            fig.add_trace(
                go.Scatter(
                    x=seg["display_time"],
                    y=seg["senkou_a"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=seg["display_time"],
                    y=seg["senkou_b"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fillcolor,
                    name=name if first_segment else None,
                    showlegend=first_segment,
                    hoverinfo="skip",
                )
            )
            first_segment = False

    bullish_mask = cloud_df["senkou_a"] >= cloud_df["senkou_b"]
    bearish_mask = cloud_df["senkou_a"] < cloud_df["senkou_b"]

    add_cloud_segments(bullish_mask, "rgba(0, 200, 0, 0.18)", "Bullish Cloud")
    add_cloud_segments(bearish_mask, "rgba(220, 0, 0, 0.18)", "Bearish Cloud")

    x_min = plot_df["display_time"].min()
    x_max = plot_df["display_time"].max() + pd.Timedelta(days=26)

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
    fig.update_xaxes(range=[x_min, x_max])

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

hourly_chart = hourly.tail(48).copy()
m15_chart = m15.tail(96).copy()
daily_chart = daily.tail(220).copy()


if not snapshots.empty:
    st.subheader("Latest signal snapshot")
    snap_view = snapshots.copy()

    if "last_open_time" in snap_view.columns:
        snap_view["last_open_time"] = snap_view["last_open_time"].apply(_fmt_display_time)
    if "updated_at" in snap_view.columns:
        snap_view["updated_at"] = snap_view["updated_at"].apply(_fmt_display_time)

    preferred_cols = [
        "pair",
        "timeframe",
        "last_open_time",
        "close",
        "ema4",
        "ema16",
        "ema65",
        "ema120",
        "rsi14",
        "rsi52",
        "signal",
        "signal_reason",
        "tenkan",
        "kijun",
        "senkou_a",
        "senkou_b",
        "chikou",
        "ichimoku_lagging_span",
        "ichimoku_conversion_vs_base",
        "ichimoku_cloud_position",
        "ichimoku_future_cloud",
        "updated_at",
    ]

    visible_cols = [c for c in preferred_cols if c in snap_view.columns]
    snap_view = snap_view[visible_cols].sort_values("timeframe")

    st.dataframe(
        snap_view,
        use_container_width=True,
    )

st.subheader(f"{selected_pair} — 1H (son 2 gün)")
st.plotly_chart(
    make_price_ema_chart(hourly_chart, f"{selected_pair} 1H — Price + EMA4/16/65/120"),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 15M (son 1 gün)")
st.plotly_chart(
    make_price_ema_chart(m15_chart, f"{selected_pair} 15M — Price + EMA4/16/65/120"),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 1D Ichimoku")
st.plotly_chart(
    make_ichimoku_chart(daily_chart, f"{selected_pair} 1D — Ichimoku"),
    use_container_width=True,
)
