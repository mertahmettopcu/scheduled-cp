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

@st.cache_data(ttl=300)
def load_pair_options() -> list[str]:
    symbols_csv_url = st.secrets["symbols"]["symbols_csv_url"]
    df = pd.read_csv(symbols_csv_url)

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "symbol" not in df.columns or "enabled" not in df.columns:
        st.error("Symbols sheet must contain 'symbol' and 'enabled' columns.")
        st.stop()

    enabled_mask = (
        df["enabled"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )

    symbols = (
        df.loc[enabled_mask, "symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    pair_options = sorted(set(symbols.tolist()))

    if not pair_options:
        st.error("Symbols sheet içinde enabled=TRUE olan symbol bulunamadı.")
        st.stop()

    return pair_options
    
params = st.query_params

if params.get("ping") == st.secrets["access"]["keepalive_token"]:
    st.write("ok")
    st.stop()
    

# =========================================================
# Config / Supabase
# =========================================================
#PAIR_OPTIONS = ["BTCUSDT", "PAXGUSDT"]
DISPLAY_TZ = "Europe/Istanbul"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "toImage",
        "select2d",
        "lasso2d",
    ],
}
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
# Auth / Access Control
# =========================================================
APP_URL = "https://scheduled-cp-xgdbr6apphni2zwxayljkbl.streamlit.app"


def _get_user_email_from_session() -> str | None:
    access_token = st.session_state.get("sb_access_token")
    if not access_token:
        return None

    try:
        user_response = supabase.auth.get_user(access_token)
        user = getattr(user_response, "user", None)
        email = getattr(user, "email", None)
        return email.lower().strip() if email else None
    except Exception:
        st.session_state.pop("sb_access_token", None)
        st.session_state.pop("sb_refresh_token", None)
        return None


params = st.query_params

if params.get("ping") == st.secrets["access"]["keepalive_token"]:
    st.write("ok")
    st.stop()

if "code" in params and "sb_access_token" not in st.session_state:
    try:
        session_response = supabase.auth.exchange_code_for_session(
            {"auth_code": params["code"]}
        )

        session = getattr(session_response, "session", None)
        if session is None:
            st.error("Login session could not be created.")
            st.stop()

        st.session_state["sb_access_token"] = session.access_token
        st.session_state["sb_refresh_token"] = session.refresh_token

        st.query_params.clear()
        st.rerun()

    except Exception:
        st.error("Google login tamamlanamadı. Lütfen tekrar deneyin.")
        st.stop()

user_email = _get_user_email_from_session()

if not user_email:
    st.title("Giriş gerekli")

    try:
        oauth_response = supabase.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {
                    "redirect_to": APP_URL,
                },
            }
        )

        login_url = getattr(oauth_response, "url", None)

        if not login_url:
            st.error("Google login URL oluşturulamadı.")
            st.stop()

        st.link_button("Google ile giriş yap", login_url)
        #st.markdown(
         #   f'<a href="{login_url}" target="_self"><button style="background-color:#FF4B4B;color:white;border:none;padding:0.5rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:1rem;">Google ile giriş yap</button></a>',
          #  unsafe_allow_html=True,
           # )

    except Exception:
        st.error("Google login başlatılamadı.")
        st.stop()

    st.stop()

allowed_emails = load_allowed_emails()

if user_email not in allowed_emails:
    st.error("Erişim izniniz yok")
    if st.button("Çıkış yap"):
        st.session_state.pop("sb_access_token", None)
        st.session_state.pop("sb_refresh_token", None)
        st.rerun()
    st.stop()

if st.button("Çıkış yap"):
    st.session_state.pop("sb_access_token", None)
    st.session_state.pop("sb_refresh_token", None)
    st.rerun()

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

def add_ichimoku_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("open_time").reset_index(drop=True)

    signals = []

    for i, row in out.iterrows():
        if i < 26:
            signals.append("NEUTRAL")
            continue

        ref = out.iloc[i - 26]

        lag_above = row["close"] > ref["high"]
        lag_below = row["close"] < ref["low"]

        conv_gt_base = row["tenkan"] > row["kijun"]
        conv_lt_base = row["tenkan"] < row["kijun"]

        a = row.get("senkou_a")
        b = row.get("senkou_b")
        close_ = row["close"]

        if pd.isna(a) or pd.isna(b):
            signals.append("NEUTRAL")
            continue

        cloud_top = max(a, b)
        cloud_bottom = min(a, b)

        above_cloud = close_ > cloud_top
        below_cloud = close_ < cloud_bottom

        future_a = row.get("senkou_a_base")
        future_b = row.get("senkou_b_base")

        future_green = pd.notna(future_a) and pd.notna(future_b) and future_a > future_b
        future_red = pd.notna(future_a) and pd.notna(future_b) and future_a < future_b

        if lag_above and conv_gt_base and above_cloud and future_green:
            signals.append("LONG")
        elif lag_below and conv_lt_base and below_cloud and future_red:
            signals.append("SHORT")
        else:
            signals.append("NEUTRAL")

    out["ichimoku_signal"] = signals
    prev_signal = out["ichimoku_signal"].shift(1).fillna("NEUTRAL")

    out["ichimoku_long_signal"] = (
        (out["ichimoku_signal"] == "LONG") &
        (prev_signal != "LONG")
    )

    out["ichimoku_short_signal"] = (
        (out["ichimoku_signal"] == "SHORT") &
        (prev_signal != "SHORT")
    )

    return out

def _rsi_from_close_series(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _sma_signal_condition_for_temp_1h_close(
    hourly_df: pd.DataFrame,
    hour_index: int,
    temp_close: float,
    signal_type: str,
) -> bool:
    if hour_index < 52:
        return False

    temp = hourly_df.iloc[: hour_index + 1].copy()
    temp.loc[temp.index[-1], "close"] = float(temp_close)

    close = pd.to_numeric(temp["close"], errors="coerce")

    sma4 = close.rolling(4).mean()
    sma16 = close.rolling(16).mean()

    rsi14 = _rsi_from_close_series(close, 14)
    rsi52 = _rsi_from_close_series(close, 52)

    last = temp.index[-1]
    prev = temp.index[-2]

    if signal_type == "LONG":
        return bool(
            (sma4.loc[last] > sma16.loc[last]) and
            (sma4.loc[prev] <= sma16.loc[prev]) and
            (rsi14.loc[last] > rsi52.loc[last]) and
            (rsi14.loc[last] >= 50)
        )

    if signal_type == "SHORT":
        return bool(
            (sma4.loc[last] < sma16.loc[last]) and
            (sma4.loc[prev] >= sma16.loc[prev]) and
            (rsi14.loc[last] < rsi52.loc[last]) and
            (rsi14.loc[last] <= 50)
        )

    return False


def build_15m_intrabar_reference_markers(
    hourly_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    only_latest: bool = True,
) -> pd.DataFrame:
    if hourly_df.empty or m15_df.empty:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    hourly_work = hourly_df.copy().sort_values("open_time").reset_index(drop=True)
    m15_work = m15_df.copy().sort_values("open_time").reset_index(drop=True)

    hourly_work["open_time"] = pd.to_datetime(hourly_work["open_time"], errors="coerce", utc=True)
    m15_work["open_time"] = pd.to_datetime(m15_work["open_time"], errors="coerce", utc=True)

    signal_rows = []

    for hour_index, row in hourly_work.iterrows():
        signal_type = None

        if bool(row.get("long_signal", False)):
            signal_type = "LONG"
        elif bool(row.get("short_signal", False)):
            signal_type = "SHORT"

        if signal_type is None:
            continue

        hour_start = row["open_time"]
        hour_end = hour_start + pd.Timedelta(hours=1)

        inside_15m = m15_work[
            (m15_work["open_time"] >= hour_start) &
            (m15_work["open_time"] < hour_end)
        ].copy()

        # En güvenli kural:
        # 1H sinyal mumunun içinde tam 4 adet 15M mum yoksa marker çizme.
        if len(inside_15m) != 4:
            continue

        condition_values = []

        for _, m15_row in inside_15m.iterrows():
            condition_values.append(
                _sma_signal_condition_for_temp_1h_close(
                    hourly_df=hourly_work,
                    hour_index=hour_index,
                    temp_close=float(m15_row["close"]),
                    signal_type=signal_type,
                )
            )

        # Eğer son 15M kapanışında koşul true değilse, resmi 1H sinyalle uyumsuzdur.
        # Bu durumda marker çizme.
        if not condition_values or not condition_values[-1]:
            continue

        # Son TRUE serisinin başladığı 15M mumu bul.
        start_pos = len(condition_values) - 1
        while start_pos > 0 and condition_values[start_pos - 1]:
            start_pos -= 1

        marker_time = inside_15m.iloc[start_pos]["open_time"]

        signal_rows.append(
            {
                "open_time": marker_time,
                "signal_type": signal_type,
            }
        )

    out = pd.DataFrame(signal_rows)
    if out.empty:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    out = (
        out
        .dropna(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if not only_latest:
        return out

    # Sadece grafikte gösterilecek son LONG + son SHORT marker.
    return latest_long_short_markers(
        out.assign(
            long_signal=out["signal_type"].eq("LONG"),
            short_signal=out["signal_type"].eq("SHORT"),
        ),
        long_col="long_signal",
        short_col="short_signal",
    )

def add_signal_reference_state(
    chart_df: pd.DataFrame,
    marker_df: pd.DataFrame,
) -> pd.DataFrame:
    out = chart_df.copy()

    # Eski/çakışan kolon varsa temizle.
    # Aksi halde merge_asof reference_signal_x / reference_signal_y üretebilir.
    out = out.drop(columns=["reference_signal"], errors="ignore")

    if marker_df is None or marker_df.empty:
        out["reference_signal"] = pd.NA
        return out

    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    markers = marker_df.copy()
    markers["open_time"] = pd.to_datetime(markers["open_time"], errors="coerce", utc=True)

    markers = (
        markers
        .dropna(subset=["open_time"])
        .sort_values("open_time")
        .rename(columns={"signal_type": "reference_signal"})
        .reset_index(drop=True)
    )

    if markers.empty:
        out["reference_signal"] = pd.NA
        return out

    merged = pd.merge_asof(
        out,
        markers[["open_time", "reference_signal"]],
        on="open_time",
        direction="backward",
    )

    return merged.sort_values("open_time").reset_index(drop=True)

def add_momentum_highlights(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    zones: pd.DataFrame,
    marker_df: pd.DataFrame,
    show_momentum: bool,
    momentum_threshold_pct: float = 35.0,
) -> go.Figure:
    if not show_momentum or chart_df.empty or zones is None or zones.empty:
        return fig

    work = add_hover_zone_context(
        chart_df=chart_df,
        zones=zones,
        zone_buffer=0.0,
    )

    work = add_signal_reference_state(work, marker_df)

    if work.empty:
        return fig

    threshold = max(float(momentum_threshold_pct or 0), 0.0) / 100.0
    offset = _chart_price_offset(work, ratio=0.018)

    for i, row in work.iterrows():
        upper_zone = row.get("hover_upper_zone")
        lower_zone = row.get("hover_lower_zone")

        if pd.isna(upper_zone) or pd.isna(lower_zone):
            continue

        zone_distance = float(upper_zone) - float(lower_zone)

        if zone_distance <= 0:
            continue

        body_size = abs(float(row["close"]) - float(row["open"]))
        momentum_ratio = body_size / zone_distance

        if momentum_ratio < threshold:
            continue

        candle_direction = "LONG" if row["close"] > row["open"] else "SHORT" if row["close"] < row["open"] else "NEUTRAL"
        raw_reference_signal = row.get("reference_signal")
        reference_signal = "" if pd.isna(raw_reference_signal) else str(raw_reference_signal).upper()

        is_counter = (
            (reference_signal == "LONG" and candle_direction == "SHORT") or
            (reference_signal == "SHORT" and candle_direction == "LONG")
        )

        display_time = row["display_time"]

        if len(work) >= 2:
            candle_delta = work["display_time"].sort_values().diff().dropna().median()
        else:
            candle_delta = pd.Timedelta(minutes=15)
        
        half_delta = candle_delta / 2
        
        fig.add_vrect(
            x0=display_time - half_delta,
            x1=display_time + half_delta,
            fillcolor="rgba(170, 215, 195, 0.30)" if not is_counter else "rgba(255, 120, 120, 0.28)",
            line_width=0,
            layer="below",
        )

    return fig

def all_signal_events(
    df: pd.DataFrame,
    long_col: str = "long_signal",
    short_col: str = "short_signal",
) -> pd.DataFrame:
    if df.empty or "open_time" not in df.columns:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    rows = []

    if long_col in df.columns:
        for _, row in df[df[long_col] == True].iterrows():
            rows.append(
                {
                    "open_time": row["open_time"],
                    "signal_type": "LONG",
                }
            )

    if short_col in df.columns:
        for _, row in df[df[short_col] == True].iterrows():
            rows.append(
                {
                    "open_time": row["open_time"],
                    "signal_type": "SHORT",
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce", utc=True)

    return (
        out
        .dropna(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

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
        .limit(1500)
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

@st.cache_data(ttl=60)
def load_manual_zones(pair: str) -> pd.DataFrame:
    r = (
        supabase.table("manual_zones")
        .select("*")
        .eq("symbol", pair)
        .eq("active", True)
        .order("sort_order", desc=False)
        .execute()
    )

    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    if "zone_value" in df.columns:
        df["zone_value"] = pd.to_numeric(df["zone_value"], errors="coerce")

    if "sort_order" in df.columns:
        df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce")

    df = (
        df
        .dropna(subset=["zone_value"])
        .sort_values("zone_value", ascending=False)
        .reset_index(drop=True)
    )

    return df
    
# =========================================================
# Indicator helpers
# =========================================================
def ema_with_sma_seed(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    result = pd.Series(index=series.index, dtype="float64")

    if len(series) < length:
        return result

    first_ema_position = length - 1
    result.iloc[first_ema_position] = series.iloc[:length].mean()

    for i in range(length, len(series)):
        result.iloc[i] = (
            alpha * series.iloc[i]
            + (1 - alpha) * result.iloc[i - 1]
        )

    return result


def add_ema_rsi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "open_time" in out.columns:
        out = out.sort_values("open_time").reset_index(drop=True)

    for length in (4, 16, 65, 120, 168):
        out[f"ema{length}"] = out["close"].ewm(span=length, adjust=False).mean()

    for length in (4, 16, 65, 120, 168):
        out[f"sma{length}"] = out["close"].rolling(length).mean()

    # Temporary debug columns for Binance comparison
    out["ema168_adjust_false"] = out["close"].ewm(span=168, adjust=False).mean()
    out["ema168_adjust_true"] = out["close"].ewm(span=168, adjust=True).mean()
    out["sma168"] = out["close"].rolling(168).mean()
    out["ema168_sma_seed"] = ema_with_sma_seed(out["close"], 168)

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

    prev_sma4 = out["sma4"].shift(1)
    prev_sma16 = out["sma16"].shift(1)
    
    out["long_signal"] = (
        (out["sma4"] > out["sma16"]) &
        (prev_sma4 <= prev_sma16) &
        (out["rsi14"] > out["rsi52"]) &
        (out["rsi14"] >= 50)
    )
    
    out["short_signal"] = (
        (out["sma4"] < out["sma16"]) &
        (prev_sma4 >= prev_sma16) &
        (out["rsi14"] < out["rsi52"]) &
        (out["rsi14"] <= 50)
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
def add_hover_zone_context(
    chart_df: pd.DataFrame,
    zones: pd.DataFrame,
    zone_buffer: float = 0.0,
) -> pd.DataFrame:
    out = chart_df.copy()

    hover_cols = [
        "hover_upper_zone",
        "hover_lower_zone",
        "hover_upper_zone_minus_buffer",
        "hover_lower_zone_plus_buffer",
    ]

    # Aynı kolonlar daha önce oluştuysa temizle.
    # Böylece customdata tarafında duplicate kolon hatası oluşmaz.
    out = out.drop(columns=[col for col in hover_cols if col in out.columns], errors="ignore")

    if zones is None or zones.empty or "close" not in out.columns:
        for col in hover_cols:
            out[col] = pd.NA
        return out

    work = zones.copy()
    work["zone_value"] = pd.to_numeric(work["zone_value"], errors="coerce")
    zone_values = (
        work["zone_value"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )

    if not zone_values:
        for col in hover_cols:
            out[col] = pd.NA
        return out

    buffer_value = max(float(zone_buffer or 0), 0.0)

    def _zone_context(close_price):
        if pd.isna(close_price):
            return pd.Series(
                {
                    "hover_upper_zone": pd.NA,
                    "hover_lower_zone": pd.NA,
                    "hover_upper_zone_minus_buffer": pd.NA,
                    "hover_lower_zone_plus_buffer": pd.NA,
                }
            )

        close_price = float(close_price)

        upper_candidates = [z for z in zone_values if z > close_price]
        lower_candidates = [z for z in zone_values if z < close_price]

        upper_zone = min(upper_candidates) if upper_candidates else pd.NA
        lower_zone = max(lower_candidates) if lower_candidates else pd.NA

        upper_minus_buffer = (
            float(upper_zone) - buffer_value
            if not pd.isna(upper_zone)
            else pd.NA
        )

        lower_plus_buffer = (
            float(lower_zone) + buffer_value
            if not pd.isna(lower_zone)
            else pd.NA
        )

        return pd.Series(
            {
                "hover_upper_zone": upper_zone,
                "hover_lower_zone": lower_zone,
                "hover_upper_zone_minus_buffer": upper_minus_buffer,
                "hover_lower_zone_plus_buffer": lower_plus_buffer,
            }
        )

    zone_context = out["close"].apply(_zone_context)

    out = pd.concat([out, zone_context], axis=1)

    return out
    
def filter_zones_for_visible_price_range(
    chart_df: pd.DataFrame,
    zones: pd.DataFrame,
) -> pd.DataFrame:
    if chart_df.empty or zones is None or zones.empty:
        return pd.DataFrame()

    if "low" not in chart_df.columns or "high" not in chart_df.columns:
        return pd.DataFrame()

    visible_low = pd.to_numeric(chart_df["low"], errors="coerce").min()
    visible_high = pd.to_numeric(chart_df["high"], errors="coerce").max()

    if pd.isna(visible_low) or pd.isna(visible_high):
        return pd.DataFrame()

    work = zones.copy()
    work["zone_value"] = pd.to_numeric(work["zone_value"], errors="coerce")
    work = work.dropna(subset=["zone_value"])

    if work.empty:
        return work

    zones_inside_range = work[
        (work["zone_value"] >= visible_low)
        & (work["zone_value"] <= visible_high)
    ]

    nearest_zone_above = (
        work[work["zone_value"] > visible_high]
        .sort_values("zone_value", ascending=True)
        .head(1)
    )

    nearest_zone_below = (
        work[work["zone_value"] < visible_low]
        .sort_values("zone_value", ascending=False)
        .head(1)
    )

    selected = pd.concat(
        [nearest_zone_above, zones_inside_range, nearest_zone_below],
        ignore_index=True,
    )

    selected = (
        selected
        .drop_duplicates(subset=["zone_value"])
        .sort_values("zone_value", ascending=False)
        .reset_index(drop=True)
    )

    return selected


def add_manual_zone_lines(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    zones: pd.DataFrame,
    show_zones: bool,
    zone_buffer: float = 0.0,
) -> go.Figure:
    if not show_zones or zones is None or zones.empty:
        return fig

    selected_zones = filter_zones_for_visible_price_range(chart_df, zones)

    if selected_zones.empty:
        return fig

    buffer_value = max(float(zone_buffer or 0), 0.0)

    for _, zone in selected_zones.iterrows():
        zone_value = zone.get("zone_value")

        if pd.isna(zone_value):
            continue

        zone_value = float(zone_value)

        note = zone.get("note")
        label = f"Zone {zone_value:.2f}"

        if note is not None and str(note).strip() and str(note).lower() != "nan":
            label = f"{label} — {str(note).strip()}"

        if buffer_value > 0:
            fig.add_hrect(
                y0=zone_value - buffer_value,
                y1=zone_value + buffer_value,
                fillcolor="rgba(128, 128, 128, 0.10)",
                line_width=0,
                layer="below",
            )

        fig.add_hline(
            y=zone_value,
            line_width=1,
            line_dash="dot",
            annotation_text=label,
            annotation_position="right",
        )

    return fig

def _chart_price_offset(chart_df: pd.DataFrame, ratio: float = 0.012) -> float:
    if chart_df.empty or "high" not in chart_df.columns or "low" not in chart_df.columns:
        return 0.0

    visible_high = pd.to_numeric(chart_df["high"], errors="coerce").max()
    visible_low = pd.to_numeric(chart_df["low"], errors="coerce").min()

    if pd.isna(visible_high) or pd.isna(visible_low):
        return 0.0

    return float(visible_high - visible_low) * ratio


def latest_long_short_markers(
    df: pd.DataFrame,
    long_col: str = "long_signal",
    short_col: str = "short_signal",
) -> pd.DataFrame:
    if df.empty or "open_time" not in df.columns:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    rows = []

    if long_col in df.columns:
        long_rows = df[df[long_col] == True]
        if not long_rows.empty:
            last_long = long_rows.iloc[-1]
            rows.append(
                {
                    "open_time": last_long["open_time"],
                    "signal_type": "LONG",
                }
            )

    if short_col in df.columns:
        short_rows = df[df[short_col] == True]
        if not short_rows.empty:
            last_short = short_rows.iloc[-1]
            rows.append(
                {
                    "open_time": last_short["open_time"],
                    "signal_type": "SHORT",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["open_time", "signal_type"])

    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce", utc=True)
    return out.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)


def add_signal_markers(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    name_prefix: str,
) -> go.Figure:
    if chart_df.empty or marker_df is None or marker_df.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)

    markers = marker_df.copy()
    markers["open_time"] = pd.to_datetime(markers["open_time"], errors="coerce", utc=True)

    merged = markers.merge(
        work[["open_time", "display_time", "high", "low"]],
        on="open_time",
        how="inner",
    )

    if merged.empty:
        return fig

    offset = _chart_price_offset(work, ratio=0.1)

    long_markers = merged[merged["signal_type"] == "LONG"].copy()
    short_markers = merged[merged["signal_type"] == "SHORT"].copy()

    if not long_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=long_markers["display_time"],
                y=long_markers["low"] - offset,
                mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="green"),
                name=f"{name_prefix} LONG",
                showlegend=False,
                customdata=long_markers[["signal_type"]],
                hovertemplate=(
                    f"{name_prefix}<br>"
                    "Signal: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if not short_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=short_markers["display_time"],
                y=short_markers["high"] + offset,
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="red"),
                name=f"{name_prefix} SHORT",
                showlegend=False,
                customdata=short_markers[["signal_type"]],
                hovertemplate=(
                    f"{name_prefix}<br>"
                    "Signal: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "<extra></extra>"
                ),
            )
        )

    return fig
    
def make_price_ema_chart(df: pd.DataFrame, title: str, zones: pd.DataFrame | None = None, show_zones: bool = False, zone_buffer: float = 0.0,ma_display: str = "EMA",signal_markers: pd.DataFrame | None = None,
    show_signal_markers: bool = False,
    signal_marker_name: str = "Signal",
    show_momentum: bool = False,
    momentum_threshold_pct: float = 35.0,
    momentum_reference_events: pd.DataFrame | None = None,) -> go.Figure:
        
    plot_df = df.copy()
    plot_df["display_time"] = plot_df["open_time"].dt.tz_convert(DISPLAY_TZ)
    
    if show_zones:
        plot_df = add_hover_zone_context(
            chart_df=plot_df,
            zones=zones,
            zone_buffer=zone_buffer,
        )
    else:
        plot_df["hover_upper_zone"] = pd.NA
        plot_df["hover_lower_zone"] = pd.NA
        plot_df["hover_upper_zone_minus_buffer"] = pd.NA
        plot_df["hover_lower_zone_plus_buffer"] = pd.NA

    fig = go.Figure()

    fig.add_trace(
    go.Candlestick(
        x=plot_df["display_time"],
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="Price",
        customdata=plot_df[
            [
                "ema168",
                "ema168_adjust_false",
                "ema168_adjust_true",
                "sma168",
                "ema168_sma_seed",
                "hover_upper_zone",
                "hover_lower_zone",
                "hover_upper_zone_minus_buffer",
                "hover_lower_zone_plus_buffer",
            ]
        ],
        hovertemplate=(
            "Time: %{x}<br>"
            "Open: %{open}<br>"
            "High: %{high}<br>"
            "Low: %{low}<br>"
            "Close: %{close}<br>"
            "<br>"
            "EMA168 current: %{customdata[0]:.4f}<br>"
            "EMA168 adjust_false: %{customdata[1]:.4f}<br>"
            "EMA168 adjust_true: %{customdata[2]:.4f}<br>"
            "SMA168: %{customdata[3]:.4f}<br>"
            "EMA168 SMA seed: %{customdata[4]:.4f}<br>"
            "<br>"
            "Manual Zone Range:<br>"
            "Upper Zone: %{customdata[5]:.2f}<br>"
            "Lower Zone: %{customdata[6]:.2f}<br>"
            "Upper Zone - Buffer: %{customdata[7]:.2f}<br>"
            "Lower Zone + Buffer: %{customdata[8]:.2f}<br>"
            "<extra></extra>"
        ),
    )
)

    ma_lengths = (4, 16, 65, 120, 168)

    if ma_display in ("EMA", "EMA + SMA"):
        for length in ma_lengths:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["display_time"],
                    y=plot_df[f"ema{length}"],
                    mode="lines",
                    name=f"EMA{length}",
                )
            )
    
    if ma_display in ("SMA", "EMA + SMA"):
        for length in ma_lengths:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["display_time"],
                    y=plot_df[f"sma{length}"],
                    mode="lines",
                    name=f"SMA{length}",
                )
            )
            
    fig = add_manual_zone_lines(
    fig=fig,
    chart_df=plot_df,
    zones=zones,
    show_zones=show_zones,
    zone_buffer=zone_buffer,
    )
    if show_signal_markers and signal_markers is not None and not signal_markers.empty:
        fig = add_signal_markers(
            fig=fig,
            chart_df=plot_df,
            marker_df=signal_markers,
            name_prefix=signal_marker_name,
        )

    fig = add_momentum_highlights(
        fig=fig,
        chart_df=plot_df,
        zones=zones,
        marker_df=momentum_reference_events if momentum_reference_events is not None else pd.DataFrame(),
        show_momentum=show_momentum,
        momentum_threshold_pct=momentum_threshold_pct,
    )
    fig.update_layout(
        title="",
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


def make_ichimoku_chart(df: pd.DataFrame, title: str, zones: pd.DataFrame | None = None, show_zones: bool = False,zone_buffer: float = 0.0, signal_markers: pd.DataFrame | None = None, show_signal_markers: bool = False,) -> go.Figure:
    plot_df = df.copy()
    plot_df["display_time"] = plot_df["open_time"].dt.tz_convert(DISPLAY_TZ)
    
    if show_zones:
        plot_df = add_hover_zone_context(
            chart_df=plot_df,
            zones=zones,
            zone_buffer=zone_buffer,
        )
    else:
        plot_df["hover_upper_zone"] = pd.NA
        plot_df["hover_lower_zone"] = pd.NA
        plot_df["hover_upper_zone_minus_buffer"] = pd.NA
        plot_df["hover_lower_zone_plus_buffer"] = pd.NA

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
                    customdata=plot_df[
                        [
                            "hover_upper_zone",
                            "hover_lower_zone",
                            "hover_upper_zone_minus_buffer",
                            "hover_lower_zone_plus_buffer",
                        ]
                    ],
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "Open: %{open}<br>"
                        "High: %{high}<br>"
                        "Low: %{low}<br>"
                        "Close: %{close}<br>"
                        "<br>"
                        "Manual Zone Range:<br>"
                        "Upper Zone: %{customdata[0]:.2f}<br>"
                        "Lower Zone: %{customdata[1]:.2f}<br>"
                        "Upper Zone - Buffer: %{customdata[2]:.2f}<br>"
                        "Lower Zone + Buffer: %{customdata[3]:.2f}<br>"
                        "<extra></extra>"
                    ),
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

    fig = add_manual_zone_lines(
    fig=fig,
    chart_df=plot_df,
    zones=zones,
    show_zones=show_zones,
    zone_buffer=zone_buffer,
)
    if show_signal_markers and signal_markers is not None and not signal_markers.empty:
        fig = add_signal_markers(
            fig=fig,
            chart_df=plot_df,
            marker_df=signal_markers,
            name_prefix="1D Ichimoku Signal",
        )
        
    x_min = plot_df["display_time"].min()
    x_max = plot_df["display_time"].max() + pd.Timedelta(days=26)

    fig.update_layout(
        title="",
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
#st.caption("Perpetual futures candles come from Supabase cache populated by your GitHub pipeline.")

#selected_pair = st.selectbox("Crypto seç", PAIR_OPTIONS, index=0)
pair_options = load_pair_options()
selected_pair = st.selectbox("Crypto seç", pair_options, index=0)
zone_buffer = st.number_input(
    "Manual zone buffer",
    min_value=0.0,
    value=150.0,
    step=50.0,
    help="Zone çizgisinin altına ve üstüne eklenecek sabit fiyat aralığı. 0 girilirse sadece zone çizgisi çizilir.",
)
ma_display = st.radio(
    "Grafikte gösterilecek hareketli ortalama",
    ["EMA", "SMA", "EMA + SMA"],
    index=1,
    horizontal=True,
    help="Bu seçim sadece grafikte çizilen ortalamaları değiştirir. Pipeline sinyali SMA4/SMA16 + RSI filtrelerine göre üretilir.",
)
momentum_threshold_pct = st.number_input(
    "Momentum threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=35.0,
    step=5.0,
    help="Momentum = abs(close - open) / zone mesafesi. Varsayılan %35. Buffer hesaba dahil edilmez.",
)

with st.expander("Gösterge açıklamaları"):
    st.markdown(
        """
- **EMA/SMA çizgileri:** Seçili hareketli ortalama çizgileri. Plotly legend'da sadece bunlar gösterilir.
- **Zone çizgisi:** Siyah kesikli yatay çizgi.
- **Zone buffer:** Zone çizgisinin altındaki/üstündeki hafif gri yatay bant.
- **Normal momentum:** Yeşil dikey gölge.
- **Counter momentum:** Kırmızı/pembe dikey gölge.
- **LONG sinyal:** Yeşil yukarı üçgen.
- **SHORT sinyal:** Kırmızı aşağı üçgen.
- **15M sinyal referansı:** 15M'in kendi sinyali değildir; 1H sinyalinin 15M içi referans noktasıdır.
        """
    )

snapshots = load_signal_snapshots(selected_pair)
hourly = load_candles(selected_pair, "1h")
m15 = load_candles(selected_pair, "15m")
daily = load_candles(selected_pair, "1d")
manual_zones = load_manual_zones(selected_pair)

if manual_zones.empty:
    st.info("Bu parite için aktif manual zone bulunamadı.")
else:
    st.caption(f"Aktif manual zone sayısı: {len(manual_zones)}")

if hourly.empty or m15.empty or daily.empty:
    st.warning("Bu coin için gerekli candle verileri Supabase'te yok. Önce pipeline'ı çalıştır.")
    st.stop()

hourly = add_ema_rsi(hourly)
m15 = add_ema_rsi(m15)
daily = add_ichimoku(daily)

hourly_chart = hourly.tail(48).copy()
m15_chart = m15.tail(96).copy()
#daily_chart = daily.tail(220).copy()

daily = add_ichimoku_signal_columns(daily)
daily_chart = daily.tail(220).copy()

hourly_signal_reference_events = all_signal_events(hourly)
hourly_signal_markers = hourly_signal_reference_events[hourly_signal_reference_events["open_time"].isin(hourly_chart["open_time"])].copy()

m15_intrabar_signal_reference_events = build_15m_intrabar_reference_markers(
    hourly_df=hourly,
    m15_df=m15,
    only_latest=False,
)
m15_intrabar_signal_markers = m15_intrabar_signal_reference_events[
    m15_intrabar_signal_reference_events["open_time"].isin(m15_chart["open_time"])
].copy()

daily_signal_reference_events = all_signal_events(
    daily,
    long_col="ichimoku_long_signal",
    short_col="ichimoku_short_signal",
)

daily_signal_markers = daily_signal_reference_events[
    daily_signal_reference_events["open_time"].isin(daily_chart["open_time"])
].copy()

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
        "sma4",
        "sma16",
        "ema4",
        "ema16",
        "ema65",
        "ema120",
        "ema168",
        "ema168_adjust_false",
        "ema168_adjust_true",
        "sma168",
        "ema168_sma_seed",
        "ema_calc_candle_count",
        "ema_calc_first_open_time",
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

col_1h_a, col_1h_b, col_1h_c = st.columns(3)

with col_1h_a:
    show_zones_1h = st.toggle("1H yakın zone çizgileri", value=True, key="show_zones_1h")

with col_1h_b:
    show_signal_markers_1h = st.toggle("1H sinyal markerları", value=True, key="show_signal_markers_1h")

with col_1h_c:
    show_momentum_1h = st.toggle("1H momentum mumları", value=True, key="show_momentum_1h")

st.plotly_chart(
    make_price_ema_chart(
        hourly_chart,
        f"{selected_pair} 1H — Price + Moving Averages",
        zones=manual_zones,
        show_zones=show_zones_1h,
        zone_buffer=zone_buffer,
        ma_display=ma_display,
        signal_markers=hourly_signal_markers,
        show_signal_markers=show_signal_markers_1h,
        signal_marker_name="1H SMA Pipeline Signal",
        show_momentum=show_momentum_1h,
        momentum_threshold_pct=momentum_threshold_pct,
        momentum_reference_events=hourly_signal_reference_events,
    ),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 15M (son 1 gün)")

col_15m_a, col_15m_b, col_15m_c = st.columns(3)

with col_15m_a:
    show_zones_15m = st.toggle("15M yakın zone çizgileri", value=True, key="show_zones_15m")

with col_15m_b:
    show_signal_markers_15m = st.toggle("15M 1H sinyal referansı", value=True, key="show_signal_markers_15m")

with col_15m_c:
    show_momentum_15m = st.toggle("15M momentum mumları", value=True, key="show_momentum_15m")

st.plotly_chart(
    make_price_ema_chart(
        m15_chart,
        f"{selected_pair} 15M — Price + Moving Averages",
        zones=manual_zones,
        show_zones=show_zones_15m,
        zone_buffer=zone_buffer,
        ma_display=ma_display,
        signal_markers=m15_intrabar_signal_markers,
        show_signal_markers=show_signal_markers_15m,
        signal_marker_name="1H Signal Intrabar Reference",
        show_momentum=show_momentum_15m,
        momentum_threshold_pct=momentum_threshold_pct,
        momentum_reference_events=m15_intrabar_signal_reference_events,
    ),
    use_container_width=True,
)

st.subheader(f"{selected_pair} — 1D Ichimoku")

col_1d_a, col_1d_b = st.columns(2)

with col_1d_a:
    show_zones_1d = st.toggle("1D yakın zone çizgileri", value=False, key="show_zones_1d")

with col_1d_b:
    show_signal_markers_1d = st.toggle("1D Ichimoku sinyal markerları", value=True, key="show_signal_markers_1d")

st.plotly_chart(
    make_ichimoku_chart(
        daily_chart,
        f"{selected_pair} 1D — Ichimoku",
        zones=manual_zones,
        show_zones=show_zones_1d,
        zone_buffer=zone_buffer,
        signal_markers=daily_signal_markers,
        show_signal_markers=show_signal_markers_1d,
    ),
    use_container_width=True,
)
