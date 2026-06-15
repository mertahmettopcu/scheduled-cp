import json
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
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

/* Sayfanın üst boşluğunu azaltır */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Başlık ile ilk input alanı arasındaki boşluğu biraz azaltır */
h2 {
    margin-bottom: 0.4rem;
}

/* Expander'ların üst-alt boşluğunu biraz kompakt hale getirir */
[data-testid="stExpander"] {
    margin-bottom: 0.4rem;
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


def _to_utc_datetime_series(values):
    try:
        return pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
            format="mixed",
        )
    except TypeError:
        return pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )


def _safe_round(value, digits=4):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def get_closed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline ile aynı resmi sinyal standardı:
    official signal / marker hesapları sadece kapanmış mumlardan üretilir.
    Açık mum grafikte görünmeye devam eder; ama resmi 1H marker üretmez.
    """
    if df.empty or "close_time" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)

    now_utc = pd.Timestamp.now(tz="UTC")

    work = work[
        work["close_time"].notna()
        & (work["close_time"] <= now_utc)
    ].copy()

    if "open_time" in work.columns:
        work = work.sort_values("open_time").reset_index(drop=True)

    return work


def add_ichimoku_signal_columns(
    df: pd.DataFrame,
    ichimoku_rr_multiplier: float = 1.7,
) -> pd.DataFrame:
    out = df.copy().sort_values("open_time").reset_index(drop=True)
    rr_multiplier = max(float(ichimoku_rr_multiplier or 0), 0.0)

    signals = []
    cloud_top_values = []
    cloud_bottom_values = []

    for i, row in out.iterrows():
        cloud_top_values.append(pd.NA)
        cloud_bottom_values.append(pd.NA)

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
        cloud_top_values[-1] = cloud_top
        cloud_bottom_values[-1] = cloud_bottom

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
    out["ichimoku_cloud_top"] = cloud_top_values
    out["ichimoku_cloud_bottom"] = cloud_bottom_values

    prev_signal = out["ichimoku_signal"].shift(1).fillna("NEUTRAL")

    out["ichimoku_long_signal"] = (
        (out["ichimoku_signal"] == "LONG") &
        (prev_signal != "LONG")
    )

    out["ichimoku_short_signal"] = (
        (out["ichimoku_signal"] == "SHORT") &
        (prev_signal != "SHORT")
    )

    # TP line confirmation rule:
    # Signal marker still appears on the signal candle.
    # TP line is drawn only if the next daily candle closes with the same Ichimoku state.
    next_signal = out["ichimoku_signal"].shift(-1)
    out["ichimoku_tp_confirmed"] = (
        (out["ichimoku_long_signal"] & next_signal.eq("LONG")) |
        (out["ichimoku_short_signal"] & next_signal.eq("SHORT"))
    )

    out["ichimoku_entry_ref"] = pd.NA
    out["ichimoku_sl_level"] = pd.NA
    out["ichimoku_sl_distance"] = pd.NA
    out["ichimoku_tp_level"] = pd.NA
    out["ichimoku_rr"] = pd.NA

    long_mask = out["ichimoku_long_signal"] == True
    short_mask = out["ichimoku_short_signal"] == True
    confirmed_mask = out["ichimoku_tp_confirmed"] == True

    # LONG: SL = cloud bottom, TP = close + RR multiplier * (close - cloud bottom)
    # TP values are populated only when the next daily candle confirms the same signal state.
    confirmed_long_mask = long_mask & confirmed_mask
    long_distance = out.loc[confirmed_long_mask, "close"] - out.loc[confirmed_long_mask, "ichimoku_cloud_bottom"]
    valid_long = confirmed_long_mask.copy()
    valid_long.loc[confirmed_long_mask] = long_distance > 0

    out.loc[valid_long, "ichimoku_entry_ref"] = out.loc[valid_long, "close"]
    out.loc[valid_long, "ichimoku_sl_level"] = out.loc[valid_long, "ichimoku_cloud_bottom"]
    out.loc[valid_long, "ichimoku_sl_distance"] = (
        out.loc[valid_long, "close"] - out.loc[valid_long, "ichimoku_cloud_bottom"]
    )
    out.loc[valid_long, "ichimoku_tp_level"] = (
        out.loc[valid_long, "close"] + rr_multiplier * out.loc[valid_long, "ichimoku_sl_distance"]
    )
    out.loc[valid_long, "ichimoku_rr"] = rr_multiplier

    # SHORT: SL = cloud top, TP = close - RR multiplier * (cloud top - close)
    # TP values are populated only when the next daily candle confirms the same signal state.
    confirmed_short_mask = short_mask & confirmed_mask
    short_distance = out.loc[confirmed_short_mask, "ichimoku_cloud_top"] - out.loc[confirmed_short_mask, "close"]
    valid_short = confirmed_short_mask.copy()
    valid_short.loc[confirmed_short_mask] = short_distance > 0

    out.loc[valid_short, "ichimoku_entry_ref"] = out.loc[valid_short, "close"]
    out.loc[valid_short, "ichimoku_sl_level"] = out.loc[valid_short, "ichimoku_cloud_top"]
    out.loc[valid_short, "ichimoku_sl_distance"] = (
        out.loc[valid_short, "ichimoku_cloud_top"] - out.loc[valid_short, "close"]
    )
    out.loc[valid_short, "ichimoku_tp_level"] = (
        out.loc[valid_short, "close"] - rr_multiplier * out.loc[valid_short, "ichimoku_sl_distance"]
    )
    out.loc[valid_short, "ichimoku_rr"] = rr_multiplier

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

    if signal_type == "LONG":
        return bool(
            (sma4.loc[last] > sma16.loc[last]) and
            (rsi14.loc[last] > rsi52.loc[last]) and
            (rsi14.loc[last] >= 50)
        )

    if signal_type == "SHORT":
        return bool(
            (sma4.loc[last] < sma16.loc[last]) and
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

    work = chart_df.copy()
    work = add_signal_reference_state(work, marker_df)

    if work.empty:
        return fig

    zone_work = zones.copy()
    zone_work["zone_value"] = pd.to_numeric(zone_work["zone_value"], errors="coerce")
    zone_values = (
        zone_work["zone_value"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )

    if not zone_values:
        return fig

    def _opening_zone_context(open_price):
        if pd.isna(open_price):
            return pd.NA, pd.NA

        open_price = float(open_price)
        upper_candidates = [z for z in zone_values if z > open_price]
        lower_candidates = [z for z in zone_values if z < open_price]

        upper_zone = min(upper_candidates) if upper_candidates else pd.NA
        lower_zone = max(lower_candidates) if lower_candidates else pd.NA

        return upper_zone, lower_zone

    threshold = max(float(momentum_threshold_pct or 0), 0.0) / 100.0

    for _, row in work.iterrows():
        # Momentum eşiği, mumun kapanış fiyatına göre değil,
        # mumun OPEN fiyatının bulunduğu zone aralığına göre hesaplanır.
        upper_zone, lower_zone = _opening_zone_context(row.get("open"))

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
            fillcolor="rgba(152, 255, 152, 0.18)" if not is_counter else "rgba(255, 120, 120, 0.28)",
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
    
@st.cache_data(ttl=60)
def load_counter_momentum_states(pair: str) -> pd.DataFrame:
    r = (
        supabase.table("counter_momentum_states")
        .select(
            "pair,timeframe,reference_signal,reference_signal_time,"
            "candle_open_time,candle_close_time,counter_direction,"
            "last_ratio,last_notified_ratio,last_status,warning_count,updated_at"
        )
        .eq("pair", pair)
        .eq("timeframe", "1h")
        .order("updated_at", desc=True)
        .limit(200)
        .execute()
    )

    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    for col in ["candle_open_time", "candle_close_time", "reference_signal_time", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["last_ratio", "last_notified_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "warning_count" in df.columns:
        df["warning_count"] = pd.to_numeric(df["warning_count"], errors="coerce")

    return (
        df
        .dropna(subset=["candle_open_time"])
        .sort_values("candle_open_time")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60)
def load_strategy_1h_state(pair: str) -> pd.DataFrame:
    r = (
        supabase.table("strategy_1h_states")
        .select("*")
        .eq("pair", pair)
        .limit(1)
        .execute()
    )

    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    time_cols = [
        "entry_time",
        "tp_candle_open_time",
        "pending_tp_hit_time",
        "last_processed_closed_open_time",
        "created_at",
        "updated_at",
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    numeric_cols = [
        "entry_price",
        "active_lower_zone",
        "active_upper_zone",
        "target_zone",
        "tp_raw_value",
        "tp_trigger",
        "pending_tp_exit_price",
        "pending_tp_raw_value",
        "pending_tp_trigger",
        "pending_old_target_zone",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(ttl=60)
def load_strategy_1h_events(pair: str, limit: int = 500) -> pd.DataFrame:
    r = (
        supabase.table("strategy_1h_events")
        .select("*")
        .eq("pair", pair)
        .order("event_time", desc=True)
        .limit(limit)
        .execute()
    )

    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    for col in ["event_time", "created_at"]:
        if col in df.columns:
            df[col] = _to_utc_datetime_series(df[col])

    if "event_time" in df.columns and "created_at" in df.columns:
        df["event_time"] = df["event_time"].fillna(df["created_at"])

    numeric_cols = [
        "price",
        "active_lower_zone",
        "active_upper_zone",
        "target_zone",
        "tp_raw_value",
        "tp_trigger",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("event_time").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_strategy_1h_tp_history(pair: str, limit: int = 1000) -> pd.DataFrame:
    r = (
        supabase.table("strategy_1h_tp_history")
        .select("*")
        .eq("pair", pair)
        .order("candle_open_time", desc=True)
        .limit(limit)
        .execute()
    )

    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df

    for col in ["candle_open_time", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    numeric_cols = [
        "tp_raw_value",
        "tp_trigger",
        "target_zone",
        "active_lower_zone",
        "active_upper_zone",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["trade_id", "candle_open_time"]).reset_index(drop=True)


def _details_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _details_dict(value) -> dict:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _details_value(value, key: str):
    data = _details_dict(value)
    return data.get(key)


def _short_reason_label(reason: str) -> str:
    reason = str(reason or "")
    labels = {
        "OFFICIAL_SIGNAL_WHILE_FLAT": "Resmî sinyal ile açılış",
        "NORMAL_REVERSE_SIGNAL": "Normal ters sinyal",
        "NORMAL_REVERSE_SIGNAL_RSI_APPROVED": "Normal ters sinyal + RSI4 onay",
        "NORMAL_REVERSE_SIGNAL_RSI_REJECTED": "Normal ters sinyal + RSI4 red",
        "COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_APPROVED": "Counter-momentum + RSI4 onay",
        "COUNTER_MOMENTUM_PLUS_REVERSE_SIGNAL_RSI_REJECTED": "Counter-momentum + RSI4 red",
        "TP_AFTER_TARGET_BUFFER_CLOSE": "TP sonrası target buffer",
        "TP_AFTER_SAME_DIRECTION_MOMENTUM": "TP sonrası aynı yön momentum",
        "TP_AFTER_RSI4_LONG_TO_SHORT": "LONG TP sonrası RSI4 → SHORT",
        "TP_AFTER_RSI4_SHORT_TO_LONG": "SHORT TP sonrası RSI4 → LONG",
        "TP_AFTER_OFFICIAL_SIGNAL": "TP sonrası resmî sinyal",
        "TP_ZONE": "Zone TP",
        "TP_SMA65": "SMA65 TP",
        "TP_SMA120": "SMA120 TP",
        "TP_SMA168": "SMA168 TP",
        "SMA_ORDER_NOT_VALID": "SMA sıralaması uygun değil",
        "NO_VALID_SMA": "Geçerli SMA TP yok",
        "NEAREST_VALID_SMA": "En yakın geçerli SMA TP",
        "TARGET_RANGE_UNAVAILABLE": "Target aralığı yok",
        "ZONE_CONTEXT_UNAVAILABLE": "Zone bağlamı yok",
        "POSITION_KEPT_WITHOUT_TP": "Pozisyon TP olmadan korundu",
        "NEW_POSITION_BLOCKED": "Yeni pozisyon engellendi",
    }
    return labels.get(reason, reason.replace("_", " ").title() if reason else "")


def _fmt_optional(value, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "NA"


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


# Dashboard-only indicator calculations.
# Official pipeline signals are still produced by process_and_publish.py / core logic.
# This helper prepares chart-side EMA/SMA/RSI columns for display.
# RSI4 is shown on the dashboard and is also used by the separate live 1H strategy engine.
# Official snapshot signal generation still uses SMA4/SMA16 + RSI14/RSI52 only.
# Keep this dashboard-side calculation aligned with strategy_1h.py when changing RSI logic.
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

    out["rsi4"] = _rsi(4)
    out["rsi14"] = _rsi(14)
    out["rsi52"] = _rsi(52)
    out["ema4_slope"] = out["ema4"].diff()
    out["ema16_slope"] = out["ema16"].diff()

    sma_bull = out["sma4"] > out["sma16"]
    sma_bear = out["sma4"] < out["sma16"]
    rsi_bull = (out["rsi14"] > out["rsi52"]) & (out["rsi14"] >= 50)
    rsi_bear = (out["rsi14"] < out["rsi52"]) & (out["rsi14"] <= 50)

    out["sma_rsi_signal"] = "NEUTRAL"
    out.loc[sma_bull & rsi_bull, "sma_rsi_signal"] = "LONG"
    out.loc[sma_bear & rsi_bear, "sma_rsi_signal"] = "SHORT"

    prev_signal = out["sma_rsi_signal"].shift(1).fillna("NEUTRAL")

    out["long_signal"] = (
        (out["sma_rsi_signal"] == "LONG") &
        (prev_signal != "LONG")
    )

    out["short_signal"] = (
        (out["sma_rsi_signal"] == "SHORT") &
        (prev_signal != "SHORT")
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

    # Hover zone range close'a göre değil, mumun OPEN fiyatına göre hesaplanır.
    # Bu, momentum hesabındaki referans zone mesafesiyle aynı mantığı kullanır.
    if zones is None or zones.empty or "open" not in out.columns:
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

    def _zone_context(open_price):
        if pd.isna(open_price):
            return pd.Series(
                {
                    "hover_upper_zone": pd.NA,
                    "hover_lower_zone": pd.NA,
                    "hover_upper_zone_minus_buffer": pd.NA,
                    "hover_lower_zone_plus_buffer": pd.NA,
                }
            )

        open_price = float(open_price)

        upper_candidates = [z for z in zone_values if z > open_price]
        lower_candidates = [z for z in zone_values if z < open_price]

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

    zone_context = out["open"].apply(_zone_context)

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




def _infer_candle_delta(chart_df: pd.DataFrame, fallback: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.Timedelta:
    if chart_df is None or chart_df.empty or "open_time" not in chart_df.columns:
        return fallback
    times = pd.to_datetime(chart_df["open_time"], errors="coerce", utc=True).dropna().sort_values()
    if len(times) < 2:
        return fallback
    delta = times.diff().dropna().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        return fallback
    return delta


def _infer_candle_label(chart_df: pd.DataFrame) -> str:
    delta = _infer_candle_delta(chart_df)
    minutes = int(round(delta.total_seconds() / 60))
    if minutes == 15:
        return "15M Mum"
    if minutes == 60:
        return "1H Mum"
    if minutes == 1440:
        return "1D Mum"
    if minutes < 60:
        return f"{minutes}M Mum"
    if minutes % 60 == 0 and minutes < 1440:
        return f"{minutes // 60}H Mum"
    return "Mum"


def _fmt_hover_price(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def _clean_hover_value(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def _build_price_candle_hover_text(row: pd.Series, label: str = "1H Mum") -> str:
    lines = [
        f"🕯️ {label}",
        f"Time: {_fmt_display_time(row.get('open_time')) or _clean_hover_value(row.get('display_time'))}",
        "O/H/L/C: "
        f"{_fmt_hover_price(row.get('open'))} / "
        f"{_fmt_hover_price(row.get('high'))} / "
        f"{_fmt_hover_price(row.get('low'))} / "
        f"{_fmt_hover_price(row.get('close'))}",
    ]

    lower_zone = row.get("hover_lower_zone")
    upper_zone = row.get("hover_upper_zone")
    if not pd.isna(lower_zone) and not pd.isna(upper_zone):
        lines.append(f"Zone: {_fmt_hover_price(lower_zone)} → {_fmt_hover_price(upper_zone)}")

        lower_buffer = row.get("hover_lower_zone_plus_buffer")
        upper_buffer = row.get("hover_upper_zone_minus_buffer")
        if not pd.isna(lower_buffer) and not pd.isna(upper_buffer):
            lines.append(f"Buffer: {_fmt_hover_price(lower_buffer)} / {_fmt_hover_price(upper_buffer)}")

    signal = _clean_hover_value(row.get("official_signal", "NA"))
    if signal not in {"NA", "NEUTRAL", "nan", "None"}:
        lines.append(f"Signal: {signal}")

    cm_status = _clean_hover_value(row.get("cm_status_text", "NA"))
    if cm_status not in {"NA", "no pipeline warning", "nan", "None"}:
        lines.append(f"CM: {cm_status}")
        counter_direction = _clean_hover_value(row.get("cm_counter_direction_text", "NA"))
        ratio = _clean_hover_value(row.get("cm_last_ratio_text", "NA"))
        if counter_direction != "NA":
            lines.append(f"Counter: {counter_direction}")
        if ratio != "NA":
            lines.append(f"Ratio: {ratio}")

    return "<br>".join(lines)


def add_candle_hover_capture_layer(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    label: str = "1H Mum",
    marker_size: int = 36,
) -> go.Figure:
    if chart_df.empty or not {"display_time", "high", "low"}.issubset(chart_df.columns):
        return fig

    work = chart_df.copy()
    work["hover_mid_price"] = (
        pd.to_numeric(work["high"], errors="coerce")
        + pd.to_numeric(work["low"], errors="coerce")
    ) / 2.0
    work["candle_hover_text"] = work.apply(lambda row: _build_price_candle_hover_text(row, label=label), axis=1)
    work = work.dropna(subset=["hover_mid_price"]).copy()

    if work.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=work["display_time"],
            y=work["hover_mid_price"],
            mode="markers",
            name=f"{label} hover",
            showlegend=False,
            marker=dict(
                size=marker_size,
                color="rgba(0,0,0,0.01)",
                line=dict(width=0),
            ),
            customdata=work["candle_hover_text"],
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    return fig

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

    offset = _chart_price_offset(work, ratio=0.04)

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


def add_ichimoku_signal_markers(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    name_prefix: str = "1D Ichimoku Signal",
) -> go.Figure:
    if chart_df.empty or marker_df is None or marker_df.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)

    markers = marker_df.copy()
    markers["open_time"] = pd.to_datetime(markers["open_time"], errors="coerce", utc=True)

    keep_cols = [
        "open_time",
        "signal_type",
        "ichimoku_entry_ref",
        "ichimoku_sl_level",
        "ichimoku_sl_distance",
        "ichimoku_tp_level",
        "ichimoku_rr",
        "ichimoku_tp_confirmed",
        "ichimoku_cloud_top",
        "ichimoku_cloud_bottom",
    ]
    keep_cols = [col for col in keep_cols if col in markers.columns]

    merged = markers[keep_cols].merge(
        work[["open_time", "display_time", "high", "low"]],
        on="open_time",
        how="inner",
    )

    if merged.empty:
        return fig

    offset = _chart_price_offset(work, ratio=0.04)

    custom_cols = [
        "signal_type",
        "ichimoku_entry_ref",
        "ichimoku_sl_level",
        "ichimoku_sl_distance",
        "ichimoku_tp_level",
        "ichimoku_rr",
        "ichimoku_tp_confirmed",
        "ichimoku_cloud_top",
        "ichimoku_cloud_bottom",
    ]

    for col in custom_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    hovertemplate = (
        f"{name_prefix}<br>"
        "Signal: %{customdata[0]}<br>"
        "Time: %{x}<br>"
        "<br>"
        "Entry ref close: %{customdata[1]:.2f}<br>"
        "SL level: %{customdata[2]:.2f}<br>"
        "SL distance: %{customdata[3]:.2f}<br>"
        "TP level: %{customdata[4]:.2f}<br>"
        "Risk/Reward: %{customdata[5]:.1f}R<br>"
        "TP confirmed next day: %{customdata[6]}<br>"
        "Cloud top: %{customdata[7]:.2f}<br>"
        "Cloud bottom: %{customdata[8]:.2f}<br>"
        "<extra></extra>"
    )

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
                customdata=long_markers[custom_cols],
                hovertemplate=hovertemplate,
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
                customdata=short_markers[custom_cols],
                hovertemplate=hovertemplate,
            )
        )

    return fig


def add_ichimoku_tp_segments(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    signal_events: pd.DataFrame,
    show_tp_lines: bool = True,
) -> go.Figure:
    if not show_tp_lines or chart_df.empty or signal_events is None or signal_events.empty:
        return fig

    required_cols = {"open_time", "signal_type", "ichimoku_tp_level"}
    if not required_cols.issubset(set(signal_events.columns)):
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)

    if "display_time" not in work.columns:
        work["display_time"] = work["open_time"].dt.tz_convert(DISPLAY_TZ)

    work = (
        work
        .dropna(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if work.empty:
        return fig

    events = signal_events.copy()
    events["open_time"] = pd.to_datetime(events["open_time"], errors="coerce", utc=True)
    events["ichimoku_tp_level"] = pd.to_numeric(events["ichimoku_tp_level"], errors="coerce")
    events["signal_type"] = events["signal_type"].astype(str).str.upper()

    # TP line confirmation:
    # If the column exists, draw TP lines only for signals confirmed by the next daily close.
    if "ichimoku_tp_confirmed" in events.columns:
        confirmed_mask = events["ichimoku_tp_confirmed"].fillna(False).astype(bool)
        events = events[confirmed_mask].copy()

    events = (
        events[events["signal_type"].isin(["LONG", "SHORT"])]
        .dropna(subset=["open_time", "ichimoku_tp_level"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if events.empty:
        return fig

    visible_start = work["open_time"].min()
    visible_end = work["open_time"].max()

    for idx, event in events.iterrows():
        signal_time = event["open_time"]
        signal_type = event["signal_type"]
        tp_level = float(event["ichimoku_tp_level"])

        next_opposite = events[
            (events["open_time"] > signal_time) &
            (events["signal_type"] != signal_type)
        ].head(1)

        next_opposite_time = (
            next_opposite.iloc[0]["open_time"]
            if not next_opposite.empty
            else pd.NaT
        )

        search_end = next_opposite_time if pd.notna(next_opposite_time) else visible_end

        search_window = work[
            (work["open_time"] >= signal_time) &
            (work["open_time"] <= search_end)
        ].copy()

        tp_hit_time = pd.NaT
        if not search_window.empty:
            if signal_type == "LONG":
                hit_rows = search_window[pd.to_numeric(search_window["high"], errors="coerce") >= tp_level]
            else:
                hit_rows = search_window[pd.to_numeric(search_window["low"], errors="coerce") <= tp_level]

            if not hit_rows.empty:
                tp_hit_time = hit_rows.iloc[0]["open_time"]

        end_time = visible_end
        tp_status = "open"

        if pd.notna(tp_hit_time):
            end_time = tp_hit_time
            tp_status = "hit"

        if pd.notna(next_opposite_time) and next_opposite_time < end_time:
            end_time = next_opposite_time
            tp_status = "ended by opposite signal"

        # Çizgi görünür grafik aralığıyla kesişmiyorsa çizme.
        if end_time < visible_start or signal_time > visible_end:
            continue

        x0_time = max(signal_time, visible_start)
        x1_time = min(end_time, visible_end)

        x0_display = pd.Timestamp(x0_time).tz_convert(DISPLAY_TZ)
        x1_display = pd.Timestamp(x1_time).tz_convert(DISPLAY_TZ)

        fig.add_trace(
            go.Scatter(
                x=[x0_display, x1_display],
                y=[tp_level, tp_level],
                mode="lines",
                line=dict(color="#D4A000", width=3),
                name="Ichimoku TP",
                showlegend=False,
                customdata=[[signal_type, tp_level, tp_status]],
                hovertemplate=(
                    "Ichimoku TP<br>"
                    "Signal: %{customdata[0]}<br>"
                    "TP level: %{customdata[1]:.2f}<br>"
                    "Status: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    return fig
    
def add_counter_momentum_hover_context(
    chart_df: pd.DataFrame,
    counter_momentum_states: pd.DataFrame | None,
) -> pd.DataFrame:
    out = chart_df.copy()

    cm_cols = [
        "cm_status_text",
        "cm_reference_signal_text",
        "cm_counter_direction_text",
        "cm_last_ratio_text",
        "cm_last_notified_ratio_text",
        "cm_warning_count_text",
        "cm_updated_at_text",
        "cm_official_note_text",
    ]

    out = out.drop(columns=[col for col in cm_cols if col in out.columns], errors="ignore")

    default_values = {
        "cm_status_text": "no pipeline warning",
        "cm_reference_signal_text": "NA",
        "cm_counter_direction_text": "NA",
        "cm_last_ratio_text": "NA",
        "cm_last_notified_ratio_text": "NA",
        "cm_warning_count_text": "NA",
        "cm_updated_at_text": "NA",
        "cm_official_note_text": "NA",
    }

    for col, value in default_values.items():
        out[col] = value

    if counter_momentum_states is None or counter_momentum_states.empty:
        return out

    if "open_time" not in out.columns or "candle_open_time" not in counter_momentum_states.columns:
        return out

    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce", utc=True)

    states = counter_momentum_states.copy()
    states["candle_open_time"] = pd.to_datetime(states["candle_open_time"], errors="coerce", utc=True)
    states = states.dropna(subset=["candle_open_time"]).copy()

    if states.empty:
        return out

    if "updated_at" in states.columns:
        states["updated_at"] = pd.to_datetime(states["updated_at"], errors="coerce", utc=True)
        states = states.sort_values(["candle_open_time", "updated_at"])
    else:
        states = states.sort_values("candle_open_time")

    # Aynı mum için birden fazla kayıt olursa en güncel state hover'da gösterilir.
    states = states.drop_duplicates(subset=["candle_open_time"], keep="last").copy()

    def _ratio_text(value):
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.2f}%"

    def _count_text(value):
        if value is None or pd.isna(value):
            return "NA"
        return str(int(value))

    def _time_text(value):
        if value is None or pd.isna(value):
            return "NA"
        return _fmt_display_time(value) or "NA"

    states["cm_status_text"] = states.get("last_status", pd.Series(index=states.index, dtype="object")).fillna("NA").astype(str)
    states["cm_reference_signal_text"] = states.get("reference_signal", pd.Series(index=states.index, dtype="object")).fillna("NA").astype(str)
    states["cm_counter_direction_text"] = states.get("counter_direction", pd.Series(index=states.index, dtype="object")).fillna("NA").astype(str)
    states["cm_last_ratio_text"] = states.get("last_ratio", pd.Series(index=states.index, dtype="float64")).apply(_ratio_text)
    states["cm_last_notified_ratio_text"] = states.get("last_notified_ratio", pd.Series(index=states.index, dtype="float64")).apply(_ratio_text)
    states["cm_warning_count_text"] = states.get("warning_count", pd.Series(index=states.index, dtype="float64")).apply(_count_text)
    states["cm_updated_at_text"] = states.get("updated_at", pd.Series(index=states.index, dtype="object")).apply(_time_text)
    states["cm_official_note_text"] = "Official 1H signal: not confirmed"

    keep_cols = [
        "candle_open_time",
        "cm_status_text",
        "cm_reference_signal_text",
        "cm_counter_direction_text",
        "cm_last_ratio_text",
        "cm_last_notified_ratio_text",
        "cm_warning_count_text",
        "cm_updated_at_text",
        "cm_official_note_text",
    ]

    merged = out.merge(
        states[keep_cols],
        left_on="open_time",
        right_on="candle_open_time",
        how="left",
        suffixes=("", "_state"),
    )

    for col, value in default_values.items():
        state_col = f"{col}_state"
        if state_col in merged.columns:
            merged[col] = merged[state_col].fillna(merged[col])
            merged = merged.drop(columns=[state_col])

    merged = merged.drop(columns=["candle_open_time"], errors="ignore")

    return merged




def _prepare_live_event_times(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()

    out = events.copy()
    if "event_time" in out.columns:
        out["event_time"] = _to_utc_datetime_series(out["event_time"])
    else:
        out["event_time"] = pd.NaT

    if "created_at" in out.columns:
        out["created_at"] = _to_utc_datetime_series(out["created_at"])
        out["event_time"] = out["event_time"].fillna(out["created_at"])

    out = out.dropna(subset=["event_time"]).copy()
    if out.empty:
        return out

    out["event_type"] = out.get("event_type", pd.Series(index=out.index, dtype="object")).fillna("").astype(str)
    out["direction"] = out.get("direction", pd.Series(index=out.index, dtype="object")).fillna("").astype(str).str.upper()
    out["reason_text"] = out.get("reason", pd.Series(index=out.index, dtype="object")).fillna("").astype(str)
    out["reason_label"] = out["reason_text"].apply(_short_reason_label)
    details_series = out.get("details", pd.Series(index=out.index, dtype="object"))
    out["details_text"] = details_series.apply(_details_text)
    out["rsi4_detail"] = pd.to_numeric(details_series.apply(lambda value: _details_value(value, "rsi4")), errors="coerce")
    out["rsi4_detail_text"] = out["rsi4_detail"].apply(_fmt_optional)
    out["price"] = pd.to_numeric(out.get("price"), errors="coerce")

    return out.sort_values(["event_time", "created_at" if "created_at" in out.columns else "event_time"]).copy()


def _build_position_change_events(event_work: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Pair same-time close/open events into one visual position-change marker."""
    if event_work is None or event_work.empty:
        return pd.DataFrame(), set()

    ev = event_work.copy().sort_values(["event_time", "created_at" if "created_at" in event_work.columns else "event_time"]).copy()
    closes = ev[ev["event_type"].eq("POSITION_CLOSE")].copy()
    opens = ev[ev["event_type"].eq("POSITION_OPEN")].copy()

    if closes.empty or opens.empty:
        return pd.DataFrame(), set()

    used_open_idx: set = set()
    paired_indices: set = set()
    rows = []

    for close_idx, close_row in closes.iterrows():
        candidates = opens.loc[~opens.index.isin(used_open_idx)].copy()
        if candidates.empty:
            continue

        candidates["time_diff_seconds"] = (
            candidates["event_time"] - close_row["event_time"]
        ).abs().dt.total_seconds()

        close_price = close_row.get("price")
        if pd.notna(close_price):
            candidates = candidates[
                candidates["price"].notna()
                & ((candidates["price"].astype(float) - float(close_price)).abs() <= 1e-8)
            ]

        candidates = candidates[candidates["time_diff_seconds"] <= 2.0].sort_values("time_diff_seconds")
        if candidates.empty:
            continue

        open_idx = candidates.index[0]
        open_row = candidates.loc[open_idx]
        used_open_idx.add(open_idx)
        paired_indices.update({close_idx, open_idx})

        old_direction = str(close_row.get("direction") or "NA").upper()
        new_direction = str(open_row.get("direction") or "NA").upper()
        label = f"{old_direction[:1]}→{new_direction[:1]}"

        rows.append(
            {
                "event_time": open_row["event_time"],
                "display_time": open_row["event_time"].tz_convert(DISPLAY_TZ),
                "price": float(open_row["price"]) if pd.notna(open_row.get("price")) else close_row.get("price"),
                "old_direction": old_direction,
                "new_direction": new_direction,
                "label": label,
                "reason_text": str(open_row.get("reason_text") or close_row.get("reason_text") or ""),
                "reason_label": str(open_row.get("reason_label") or close_row.get("reason_label") or ""),
                "rsi4_detail_text": str(close_row.get("rsi4_detail_text") or open_row.get("rsi4_detail_text") or "NA"),
                "close_trade_id": close_row.get("trade_id"),
                "open_trade_id": open_row.get("trade_id"),
                "close_details_text": close_row.get("details_text") or "",
                "open_details_text": open_row.get("details_text") or "",
            }
        )

    return pd.DataFrame(rows), paired_indices


def _attach_price_candle_context(events_df: pd.DataFrame, chart_work: pd.DataFrame) -> pd.DataFrame:
    """Attach the candle high/low that contains each event_time.

    Live close/open reversal events are timestamped at the candle close
    (for example 20:59:59.999). For visual clarity we place the marker
    near the candle wick instead of directly on the close price.
    """
    if events_df is None or events_df.empty or chart_work is None or chart_work.empty:
        return events_df

    events = events_df.copy()
    work = chart_work.copy()

    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)

    if "close_time" in work.columns:
        work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)
    else:
        work["close_time"] = work["open_time"] + pd.Timedelta(hours=1)

    work = work.dropna(subset=["open_time", "close_time"]).sort_values("open_time")
    events = events.dropna(subset=["event_time"]).sort_values("event_time")

    if work.empty or events.empty:
        return events_df

    keep_cols = ["open_time", "close_time", "high", "low"]
    merged = pd.merge_asof(
        events,
        work[keep_cols],
        left_on="event_time",
        right_on="open_time",
        direction="backward",
    )

    in_candle = merged["event_time"] <= merged["close_time"]
    merged.loc[~in_candle, ["high", "low"]] = pd.NA
    return merged


def add_live_strategy_event_markers(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    events: pd.DataFrame | None,
    show_markers: bool = True,
) -> go.Figure:
    if not show_markers or chart_df.empty or events is None or events.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["open_time"]).sort_values("open_time")
    if work.empty:
        return fig

    event_work = _prepare_live_event_times(events)
    if event_work.empty:
        return fig

    visible_start = work["open_time"].min()
    visible_end = work["close_time"].max() if "close_time" in work.columns else work["open_time"].max() + pd.Timedelta(hours=1)
    event_work = event_work[
        (event_work["event_time"] >= visible_start)
        & (event_work["event_time"] <= visible_end)
    ].copy()
    if event_work.empty:
        return fig

    event_work["display_time"] = event_work["event_time"].dt.tz_convert(DISPLAY_TZ)
    offset = _chart_price_offset(work, ratio=0.028)

    change_events, paired_indices = _build_position_change_events(event_work)
    if not change_events.empty:
        change_events = _attach_price_candle_context(change_events, work)
        wick_offset = _chart_price_offset(work, ratio=0.006)
        change_events["marker_y"] = change_events.apply(
            lambda row: (
                float(row["high"]) + wick_offset
                if row["new_direction"] == "SHORT" and pd.notna(row.get("high"))
                else float(row["low"]) - wick_offset
                if row["new_direction"] == "LONG" and pd.notna(row.get("low"))
                else float(row["price"]) + (offset * 1.1 if row["new_direction"] == "SHORT" else -offset * 1.1)
            ),
            axis=1,
        )
        change_events["text_position"] = change_events["new_direction"].apply(
            lambda direction: "top center" if direction == "SHORT" else "bottom center"
        )

        fig.add_trace(
            go.Scatter(
                x=change_events["display_time"],
                y=change_events["marker_y"],
                mode="markers+text",
                marker=dict(symbol="diamond", size=11, color="#F28E2B", line=dict(width=1.2, color="black")),
                text=change_events["label"],
                textposition=change_events["text_position"],
                textfont=dict(size=9, color="#222222"),
                name="Live position change",
                showlegend=False,
                customdata=change_events[[
                    "old_direction",
                    "new_direction",
                    "price",
                    "reason_label",
                    "rsi4_detail_text",
                ]],
                hovertemplate=(
                    "🟠 Pozisyon değişti<br>"
                    "%{customdata[0]} → %{customdata[1]}<br>"
                    "Fiyat: %{customdata[2]:.2f}<br>"
                    "Sebep: %{customdata[3]}<br>"
                    "RSI4: %{customdata[4]}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    unpaired_events = event_work.drop(index=list(paired_indices), errors="ignore").copy()

    opens = unpaired_events[unpaired_events["event_type"].eq("POSITION_OPEN")].copy()
    if not opens.empty:
        long_opens = opens[opens["direction"].eq("LONG")]
        short_opens = opens[opens["direction"].eq("SHORT")]

        for subset, symbol, color, y_adjust, name in [
            (long_opens, "triangle-up", "#00A65A", -offset, "Live LONG entry"),
            (short_opens, "triangle-down", "#D62728", offset, "Live SHORT entry"),
        ]:
            if subset.empty:
                continue
            y = subset["price"] + y_adjust
            fig.add_trace(
                go.Scatter(
                    x=subset["display_time"],
                    y=y,
                    mode="markers",
                    marker=dict(symbol=symbol, size=15, color=color, line=dict(width=1, color="white")),
                    name=name,
                    showlegend=False,
                    customdata=subset[["direction", "price", "reason_label", "target_zone"]],
                    hovertemplate=(
                        "🟢 Pozisyon açıldı<br>"
                        "Yön: %{customdata[0]}<br>"
                        "Entry: %{customdata[1]:.2f}<br>"
                        "Sebep: %{customdata[2]}<br>"
                        "Target: %{customdata[3]:.2f}<br>"
                        "%{x}<extra></extra>"
                    ),
                )
            )

    tp_hits = event_work[event_work["event_type"].eq("TP_HIT")].copy()
    if not tp_hits.empty:
        fig.add_trace(
            go.Scatter(
                x=tp_hits["display_time"],
                y=tp_hits["price"],
                mode="markers",
                marker=dict(symbol="star", size=14, color="#D4A000", line=dict(width=1, color="black")),
                name="Live TP hit",
                showlegend=False,
                customdata=tp_hits[["direction", "price", "reason_label", "tp_source", "tp_trigger", "target_zone"]],
                hovertemplate=(
                    "⭐ TP oldu<br>"
                    "Yön: %{customdata[0]}<br>"
                    "Exit: %{customdata[1]:.2f}<br>"
                    "TP: %{customdata[3]} @ %{customdata[4]:.2f}<br>"
                    "Target: %{customdata[5]:.2f}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    closes = unpaired_events[unpaired_events["event_type"].eq("POSITION_CLOSE")].copy()
    if not closes.empty:
        closes["marker_y"] = closes.apply(
            lambda row: float(row["price"]) + (offset * 1.2 if row["direction"] == "LONG" else -offset * 1.2),
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=closes["display_time"],
                y=closes["marker_y"],
                mode="markers",
                marker=dict(symbol="circle-x", size=13, color="#4D4D4D", line=dict(width=1.5, color="white")),
                name="Live position close",
                showlegend=False,
                customdata=closes[["direction", "price", "reason_label"]],
                hovertemplate=(
                    "⚫ Pozisyon kapandı<br>"
                    "Yön: %{customdata[0]}<br>"
                    "Exit: %{customdata[1]:.2f}<br>"
                    "Sebep: %{customdata[2]}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    held = event_work[event_work["event_type"].eq("POSITION_HELD")].copy()
    if not held.empty:
        fig.add_trace(
            go.Scatter(
                x=held["display_time"],
                y=held["price"],
                mode="markers",
                marker=dict(symbol="diamond-open", size=12, color="#9467BD", line=dict(width=2)),
                name="Position held",
                showlegend=False,
                customdata=held[["direction", "price", "reason_label", "rsi4_detail_text"]],
                hovertemplate=(
                    "🟣 Pozisyon korundu<br>"
                    "Yön: %{customdata[0]}<br>"
                    "Fiyat: %{customdata[1]:.2f}<br>"
                    "Sebep: %{customdata[2]}<br>"
                    "RSI4: %{customdata[3]}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    return fig

def add_live_strategy_tp_segments(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    tp_history: pd.DataFrame | None,
    events: pd.DataFrame | None,
    state: pd.DataFrame | None,
    show_lines: bool = True,
) -> go.Figure:
    if not show_lines or chart_df.empty or tp_history is None or tp_history.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["open_time"]).sort_values("open_time")
    if work.empty:
        return fig

    history = tp_history.copy()
    history["candle_open_time"] = pd.to_datetime(history["candle_open_time"], errors="coerce", utc=True)
    history["tp_trigger"] = pd.to_numeric(history["tp_trigger"], errors="coerce")
    history["tp_raw_value"] = pd.to_numeric(history["tp_raw_value"], errors="coerce")
    history = history.dropna(subset=["candle_open_time", "tp_trigger", "trade_id"])
    if history.empty:
        return fig

    visible_start = work["open_time"].min()
    if "close_time" in work.columns:
        work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)
        visible_end = work["close_time"].max()
    else:
        visible_end = work["open_time"].max() + _infer_candle_delta(work)

    event_work = events.copy() if events is not None else pd.DataFrame()
    if not event_work.empty:
        event_work["event_time"] = _to_utc_datetime_series(event_work["event_time"])

        if "created_at" in event_work.columns:
            event_work["created_at"] = _to_utc_datetime_series(event_work["created_at"])
            event_work["event_time"] = event_work["event_time"].fillna(event_work["created_at"])

    active_trade_id = None
    if state is not None and not state.empty and str(state.iloc[0].get("status") or "").upper() in {"OPEN", "PENDING_TP_DECISION"}:
        active_trade_id = state.iloc[0].get("trade_id")

    for trade_id, trade_hist in history.groupby("trade_id", sort=False):
        trade_hist = trade_hist.sort_values("candle_open_time").reset_index(drop=True)

        trade_end = pd.NaT
        if not event_work.empty:
            terminal = event_work[
                event_work["trade_id"].astype(str).eq(str(trade_id))
                & event_work["event_type"].isin(["TP_HIT", "POSITION_CLOSE"])
            ].sort_values("event_time")
            if not terminal.empty:
                trade_end = terminal.iloc[-1]["event_time"]

        for i, row in trade_hist.iterrows():
            start = row["candle_open_time"]
            if i + 1 < len(trade_hist):
                end = trade_hist.iloc[i + 1]["candle_open_time"]
            elif pd.notna(trade_end):
                end = trade_end
            elif str(trade_id) == str(active_trade_id):
                end = visible_end
            else:
                end = start + pd.Timedelta(hours=1)

            if end < visible_start or start > visible_end:
                continue

            x0 = max(start, visible_start).tz_convert(DISPLAY_TZ)
            x1 = min(end, visible_end).tz_convert(DISPLAY_TZ)
            trigger = float(row["tp_trigger"])
            source = str(row.get("tp_source") or "NA")

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[trigger, trigger],
                    mode="lines",
                    line=dict(color="#D4A000", width=3),
                    name="Live strategy TP",
                    showlegend=False,
                    customdata=[[
                        str(row.get("direction") or "NA"),
                        source,
                        row.get("tp_raw_value"),
                        trigger,
                        row.get("target_zone"),
                        row.get("active_lower_zone"),
                        row.get("active_upper_zone"),
                    ]] * 2,
                    hovertemplate=(
                        "🟡 TP çizgisi<br>"
                        "Yön: %{customdata[0]}<br>"
                        "TP: %{customdata[1]} @ %{customdata[3]:.2f}<br>"
                        "Target: %{customdata[4]:.2f}<br>"
                        "%{x}<extra></extra>"
                    ),
                )
            )

    return fig


def add_live_strategy_trade_segments(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    events: pd.DataFrame | None,
    state: pd.DataFrame | None,
    show_segments: bool = True,
) -> go.Figure:
    if not show_segments or chart_df.empty or events is None or events.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)
    visible_start = work["open_time"].min()
    if "close_time" in work.columns:
        work["close_time"] = pd.to_datetime(work["close_time"], errors="coerce", utc=True)
        visible_end = work["close_time"].max()
    else:
        visible_end = work["open_time"].max() + _infer_candle_delta(work)

    ev = events.copy()
    ev["event_time"] = _to_utc_datetime_series(ev["event_time"])

    if "created_at" in ev.columns:
        ev["created_at"] = _to_utc_datetime_series(ev["created_at"])
        ev["event_time"] = ev["event_time"].fillna(ev["created_at"])

    ev["price"] = pd.to_numeric(ev["price"], errors="coerce")
    ev = ev.dropna(subset=["event_time", "price", "trade_id"])

    active_trade_id = None
    if state is not None and not state.empty and str(state.iloc[0].get("status") or "").upper() == "OPEN":
        active_trade_id = state.iloc[0].get("trade_id")

    for trade_id, trade_events in ev.groupby("trade_id", sort=False):
        trade_events = trade_events.sort_values("event_time")
        opens = trade_events[trade_events["event_type"].eq("POSITION_OPEN")]
        if opens.empty:
            continue
        entry = opens.iloc[0]

        exits = trade_events[trade_events["event_type"].isin(["TP_HIT", "POSITION_CLOSE"])]
        if not exits.empty:
            exit_row = exits.iloc[-1]
            end_time = exit_row["event_time"]
            end_price = float(exit_row["price"])
            status = str(exit_row["event_type"])
        elif str(trade_id) == str(active_trade_id):
            end_time = visible_end
            end_price = float(work.iloc[-1]["close"])
            status = "OPEN"
        else:
            continue

        start_time = entry["event_time"]
        if end_time < visible_start or start_time > visible_end:
            continue

        clipped_start = max(start_time, visible_start)
        clipped_end = min(end_time, visible_end)
        entry_price = float(entry["price"])

        fig.add_trace(
            go.Scatter(
                x=[clipped_start.tz_convert(DISPLAY_TZ), clipped_end.tz_convert(DISPLAY_TZ)],
                y=[entry_price, end_price],
                mode="lines",
                line=dict(width=1.5, dash="dot"),
                name="Live trade path",
                showlegend=False,
                customdata=[[
                    str(entry.get("direction") or "NA"),
                    str(entry.get("reason") or ""),
                    entry_price,
                    end_price,
                    status,
                ]] * 2,
                hovertemplate=(
                    "Trade path<br>"
                    "%{customdata[0]} | %{customdata[4]}<br>"
                    "Entry: %{customdata[2]:.2f}<br>"
                    "Exit/current: %{customdata[3]:.2f}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    return fig

def make_price_ema_chart(
    df: pd.DataFrame,
    title: str,
    zones: pd.DataFrame | None = None,
    show_zones: bool = False,
    zone_buffer: float = 0.0,
    ma_display: str = "EMA",
    signal_markers: pd.DataFrame | None = None,
    show_signal_markers: bool = False,
    signal_marker_name: str = "Signal",
    show_momentum: bool = False,
    momentum_threshold_pct: float = 35.0,
    momentum_reference_events: pd.DataFrame | None = None,
    counter_momentum_states: pd.DataFrame | None = None,
    live_strategy_state: pd.DataFrame | None = None,
    live_strategy_events: pd.DataFrame | None = None,
    live_strategy_tp_history: pd.DataFrame | None = None,
    show_live_strategy: bool = False,
) -> go.Figure:
        
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

    plot_df = add_counter_momentum_hover_context(
        chart_df=plot_df,
        counter_momentum_states=counter_momentum_states,
    )

    fig = go.Figure()

    fig.add_trace(
    go.Candlestick(
        x=plot_df["display_time"],
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="Price",
        showlegend=False,
        hoverinfo="skip",
    )
)

    fig = add_candle_hover_capture_layer(
        fig=fig,
        chart_df=plot_df,
        label=_infer_candle_label(plot_df),
        marker_size=38,
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
                    hoverinfo="skip",
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
                    hovertemplate=f"SMA{length}: %{{y:.4f}}<extra></extra>",
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

    fig = add_live_strategy_trade_segments(
        fig=fig,
        chart_df=plot_df,
        events=live_strategy_events,
        state=live_strategy_state,
        show_segments=show_live_strategy,
    )
    fig = add_live_strategy_tp_segments(
        fig=fig,
        chart_df=plot_df,
        tp_history=live_strategy_tp_history,
        events=live_strategy_events,
        state=live_strategy_state,
        show_lines=show_live_strategy,
    )
    fig = add_live_strategy_event_markers(
        fig=fig,
        chart_df=plot_df,
        events=live_strategy_events,
        show_markers=show_live_strategy,
    )

    fig.update_layout(
        title="",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        hovermode="closest",
        hoverdistance=90,
        spikedistance=90,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
    )
    return fig


def add_rsi_signal_markers(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    name_prefix: str = "1H SMA Pipeline Signal",
) -> go.Figure:
    # RSI panel signal arrows are visual references only.
    # Official signal generation remains in the pipeline/core layer.
    if chart_df.empty or marker_df is None or marker_df.empty:
        return fig

    required_cols = {"open_time", "display_time", "rsi14"}
    if not required_cols.issubset(set(chart_df.columns)):
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)

    markers = marker_df.copy()
    markers["open_time"] = pd.to_datetime(markers["open_time"], errors="coerce", utc=True)

    keep_cols = ["open_time", "display_time", "rsi4", "rsi14", "rsi52"]
    keep_cols = [col for col in keep_cols if col in work.columns]

    merged = markers.merge(
        work[keep_cols],
        on="open_time",
        how="inner",
    ).dropna(subset=["rsi14"])

    if merged.empty:
        return fig

    long_markers = merged[merged["signal_type"] == "LONG"].copy()
    short_markers = merged[merged["signal_type"] == "SHORT"].copy()

    custom_cols = [col for col in ["signal_type", "rsi4", "rsi14", "rsi52"] if col in merged.columns]

    if not long_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=long_markers["display_time"],
                y=long_markers["rsi14"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="green"),
                name=f"{name_prefix} LONG",
                showlegend=False,
                customdata=long_markers[custom_cols],
                hovertemplate=(
                    f"{name_prefix}<br>"
                    "Signal: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "RSI4: %{customdata[1]:.2f}<br>"
                    "RSI14: %{customdata[2]:.2f}<br>"
                    "RSI52: %{customdata[3]:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if not short_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=short_markers["display_time"],
                y=short_markers["rsi14"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="red"),
                name=f"{name_prefix} SHORT",
                showlegend=False,
                customdata=short_markers[custom_cols],
                hovertemplate=(
                    f"{name_prefix}<br>"
                    "Signal: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "RSI4: %{customdata[1]:.2f}<br>"
                    "RSI14: %{customdata[2]:.2f}<br>"
                    "RSI52: %{customdata[3]:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    return fig


# 1H RSI panel shown directly under the 1H price chart.
# RSI values are calculated in add_ema_rsi(); this function renders RSI4/RSI14/RSI52.
# Signal arrows and momentum highlights are copied as visual references from the official 1H chart context.

def _events_with_rsi_context(event_df: pd.DataFrame, chart_df: pd.DataFrame) -> pd.DataFrame:
    if event_df is None or event_df.empty or chart_df.empty:
        return pd.DataFrame()

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    if work.empty or "rsi14" not in work.columns:
        return pd.DataFrame()

    keep_cols = ["open_time", "rsi4", "rsi14", "rsi52"]
    keep_cols = [col for col in keep_cols if col in work.columns]

    ev = event_df.copy().sort_values("event_time").reset_index(drop=True)
    merged = pd.merge_asof(
        ev,
        work[keep_cols].sort_values("open_time"),
        left_on="event_time",
        right_on="open_time",
        direction="backward",
    )

    merged["display_time"] = merged["event_time"].dt.tz_convert(DISPLAY_TZ)
    merged["rsi_marker_y"] = pd.to_numeric(merged.get("rsi14"), errors="coerce")
    return merged.dropna(subset=["rsi_marker_y"])


def add_live_strategy_rsi_event_markers(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    events: pd.DataFrame | None,
    show_markers: bool = True,
) -> go.Figure:
    if not show_markers or chart_df.empty or events is None or events.empty:
        return fig

    work = chart_df.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["open_time"]).sort_values("open_time")
    if work.empty:
        return fig

    event_work = _prepare_live_event_times(events)
    if event_work.empty:
        return fig

    visible_start = work["open_time"].min()
    visible_end = work["open_time"].max() + pd.Timedelta(hours=1)
    event_work = event_work[
        (event_work["event_time"] >= visible_start)
        & (event_work["event_time"] <= visible_end)
    ].copy()
    if event_work.empty:
        return fig

    change_events, paired_indices = _build_position_change_events(event_work)
    if not change_events.empty:
        rsi_changes = _events_with_rsi_context(change_events, work)
        if not rsi_changes.empty:
            fig.add_trace(
                go.Scatter(
                    x=rsi_changes["display_time"],
                    y=rsi_changes["rsi_marker_y"],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=10, color="#F28E2B", line=dict(width=1.2, color="black")),
                    text=rsi_changes["label"],
                    textposition="top center",
                    textfont=dict(size=9, color="#222222"),
                    name="Live position change RSI",
                    showlegend=False,
                    customdata=rsi_changes[[
                        "old_direction",
                        "new_direction",
                        "reason_label",
                        "rsi4",
                        "rsi14",
                        "rsi52",
                    ]],
                    hovertemplate=(
                        "🟠 Pozisyon değişti<br>"
                        "%{customdata[0]} → %{customdata[1]}<br>"
                        "Sebep: %{customdata[2]}<br>"
                        "RSI4: %{customdata[3]:.2f}<br>"
                        "RSI14/52: %{customdata[4]:.2f} / %{customdata[5]:.2f}<br>"
                        "%{x}<extra></extra>"
                    ),
                )
            )

    unpaired_events = event_work.drop(index=list(paired_indices), errors="ignore").copy()

    opens = _events_with_rsi_context(unpaired_events[unpaired_events["event_type"].eq("POSITION_OPEN")], work)
    if not opens.empty:
        long_opens = opens[opens["direction"].eq("LONG")]
        short_opens = opens[opens["direction"].eq("SHORT")]
        for subset, symbol, color, name in [
            (long_opens, "triangle-up", "#00A65A", "Live LONG entry RSI"),
            (short_opens, "triangle-down", "#D62728", "Live SHORT entry RSI"),
        ]:
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["display_time"],
                    y=subset["rsi_marker_y"],
                    mode="markers",
                    marker=dict(symbol=symbol, size=13, color=color, line=dict(width=1, color="white")),
                    name=name,
                    showlegend=False,
                    customdata=subset[["direction", "reason_label", "rsi4", "rsi14", "rsi52"]],
                    hovertemplate=(
                        "🟢 Pozisyon açıldı<br>"
                        "Yön: %{customdata[0]}<br>"
                        "Sebep: %{customdata[1]}<br>"
                        "RSI4: %{customdata[2]:.2f}<br>"
                        "RSI14/52: %{customdata[3]:.2f} / %{customdata[4]:.2f}<br>"
                        "%{x}<extra></extra>"
                    ),
                )
            )

    tp_hits = _events_with_rsi_context(event_work[event_work["event_type"].eq("TP_HIT")], work)
    if not tp_hits.empty:
        fig.add_trace(
            go.Scatter(
                x=tp_hits["display_time"],
                y=tp_hits["rsi_marker_y"],
                mode="markers",
                marker=dict(symbol="star", size=13, color="#D4A000", line=dict(width=1, color="black")),
                name="Live TP hit RSI",
                showlegend=False,
                customdata=tp_hits[["direction", "tp_source", "tp_trigger", "rsi4", "rsi14", "rsi52"]],
                hovertemplate=(
                    "⭐ TP oldu<br>"
                    "Yön: %{customdata[0]}<br>"
                    "TP: %{customdata[1]} @ %{customdata[2]:.2f}<br>"
                    "RSI4: %{customdata[3]:.2f}<br>"
                    "RSI14/52: %{customdata[4]:.2f} / %{customdata[5]:.2f}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    closes = _events_with_rsi_context(unpaired_events[unpaired_events["event_type"].eq("POSITION_CLOSE")], work)
    if not closes.empty:
        fig.add_trace(
            go.Scatter(
                x=closes["display_time"],
                y=closes["rsi_marker_y"],
                mode="markers",
                marker=dict(symbol="circle-x", size=12, color="#4D4D4D", line=dict(width=1.5, color="white")),
                name="Live position close RSI",
                showlegend=False,
                customdata=closes[["direction", "reason_label", "rsi4", "rsi14", "rsi52"]],
                hovertemplate=(
                    "⚫ Pozisyon kapandı<br>"
                    "Yön: %{customdata[0]}<br>"
                    "Sebep: %{customdata[1]}<br>"
                    "RSI4: %{customdata[2]:.2f}<br>"
                    "RSI14/52: %{customdata[3]:.2f} / %{customdata[4]:.2f}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    held = _events_with_rsi_context(event_work[event_work["event_type"].eq("POSITION_HELD")], work)
    if not held.empty:
        fig.add_trace(
            go.Scatter(
                x=held["display_time"],
                y=held["rsi_marker_y"],
                mode="markers",
                marker=dict(symbol="diamond-open", size=12, color="#9467BD", line=dict(width=2)),
                name="Position held RSI",
                showlegend=False,
                customdata=held[["direction", "reason_label", "rsi4", "rsi14", "rsi52"]],
                hovertemplate=(
                    "🟣 Pozisyon korundu<br>"
                    "Yön: %{customdata[0]}<br>"
                    "Sebep: %{customdata[1]}<br>"
                    "RSI4: %{customdata[2]:.2f}<br>"
                    "RSI14/52: %{customdata[3]:.2f} / %{customdata[4]:.2f}<br>"
                    "%{x}<extra></extra>"
                ),
            )
        )

    return fig

def make_1h_rsi_chart(
    df: pd.DataFrame,
    title: str,
    zones: pd.DataFrame | None = None,
    signal_markers: pd.DataFrame | None = None,
    show_signal_markers: bool = False,
    show_momentum: bool = False,
    momentum_threshold_pct: float = 35.0,
    momentum_reference_events: pd.DataFrame | None = None,
    live_strategy_events: pd.DataFrame | None = None,
    show_live_strategy: bool = False,
) -> go.Figure:
    plot_df = df.copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=260,
            xaxis_title="Time",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100], tickmode="array", tickvals=[20, 30, 50, 70, 80.5]),
        )
        return fig

    plot_df["display_time"] = plot_df["open_time"].dt.tz_convert(DISPLAY_TZ)

    fig = go.Figure()

    # Momentum shading uses the same open-based zone-distance logic as the 1H price chart.
    # It is drawn before RSI lines so RSI remains visually dominant.
    fig = add_momentum_highlights(
        fig=fig,
        chart_df=plot_df,
        zones=zones,
        marker_df=momentum_reference_events if momentum_reference_events is not None else pd.DataFrame(),
        show_momentum=show_momentum,
        momentum_threshold_pct=momentum_threshold_pct,
    )

    rsi_specs = (
        ("rsi4", "RSI4", "#f1c40f"),
        ("rsi14", "RSI14", "#9b59b6"),
        ("rsi52", "RSI52", "#e74c3c"),
    )

    for column, name, color in rsi_specs:
        if column not in plot_df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=plot_df["display_time"],
                y=plot_df[column],
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            )
        )

    for level in (20, 30, 50, 70, 80.5):
        fig.add_hline(
            y=level,
            line_width=1,
            line_dash="dot",
            opacity=0.35,
            annotation_text=f"{level:g}",
            annotation_position="right",
        )

    if show_signal_markers and signal_markers is not None and not signal_markers.empty:
        fig = add_rsi_signal_markers(
            fig=fig,
            chart_df=plot_df,
            marker_df=signal_markers,
            name_prefix="1H SMA Pipeline Signal",
        )

    fig = add_live_strategy_rsi_event_markers(
        fig=fig,
        chart_df=plot_df,
        events=live_strategy_events,
        show_markers=show_live_strategy,
    )

    fig.update_layout(
        title="",
        xaxis_title="Time",
        yaxis_title="RSI",
        yaxis=dict(
            range=[0, 100],
            tickmode="array",
            tickvals=[20, 30, 50, 70, 80.5],
        ),
        height=280,
        hovermode="closest",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    return fig


def make_ichimoku_chart(df: pd.DataFrame, title: str, zones: pd.DataFrame | None = None, show_zones: bool = False,zone_buffer: float = 0.0, signal_markers: pd.DataFrame | None = None, show_signal_markers: bool = False, tp_signal_events: pd.DataFrame | None = None, show_tp_lines: bool = True,) -> go.Figure:
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
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig = add_candle_hover_capture_layer(
        fig=fig,
        chart_df=plot_df,
        label="1D Mum",
        marker_size=34,
    )

    # Tenkan / Kijun / Chikou
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["tenkan"],
            mode="lines",
            name="Tenkan",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["kijun"],
            mode="lines",
            name="Kijun",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["display_time"],
            y=plot_df["chikou"],
            mode="lines",
            name="Chikou",
            showlegend=False,
        )
    )

    # Senkou outlines
    fig.add_trace(
        go.Scatter(
            x=cloud_df["display_time"],
            y=cloud_df["senkou_a"],
            mode="lines",
            name="Senkou A",
            showlegend=False,
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cloud_df["display_time"],
            y=cloud_df["senkou_b"],
            mode="lines",
            name="Senkou B",
            showlegend=False,
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
                    showlegend=False,
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
    fig = add_ichimoku_tp_segments(
        fig=fig,
        chart_df=plot_df,
        signal_events=tp_signal_events if tp_signal_events is not None else pd.DataFrame(),
        show_tp_lines=show_tp_lines,
    )

    if show_signal_markers and signal_markers is not None and not signal_markers.empty:
        fig = add_ichimoku_signal_markers(
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
        hovermode="closest",
        hoverdistance=90,
        spikedistance=90,
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
st.markdown("## Crypto Futures Strategy Dashboard")
# st.caption("Perpetual futures candles come from Supabase cache populated by your GitHub pipeline.")

refresh_count = st_autorefresh(
    interval=300000,
    key="dashboard_autorefresh",
)

st.caption(
    f"Dashboard otomatik yenilenir: 5 dk | Son yenileme: {pd.Timestamp.now(tz=DISPLAY_TZ).strftime('%Y-%m-%d %H:%M:%S')} | Refresh: {refresh_count}"
)

pair_options = load_pair_options()

top_col1, top_col2 = st.columns([1.1, 1.9])

with top_col1:
    selected_pair = st.selectbox(
        "Crypto seç",
        pair_options,
        index=0,
    )

with top_col2:
    ma_display = st.radio(
        "Grafikte gösterilecek hareketli ortalama",
        ["EMA", "SMA", "EMA + SMA"],
        index=1,
        horizontal=True,
        help="Bu seçim sadece grafikte çizilen ortalamaları değiştirir. Pipeline sinyali ve snapshot SMA bazlıdır. Snapshot değerleri dashboard içinde yeniden hesaplanmaz; Supabase’den geldiği gibi gösterilir.",
    )

with st.expander("Grafik ayarları", expanded=False):
    settings_col1, settings_col2, settings_col3 = st.columns(3)

    with settings_col1:
        zone_buffer = st.number_input(
            "Manual zone buffer",
            min_value=0.0,
            value=150.0,
            step=50.0,
            help="Zone çizgisinin altına ve üstüne eklenecek sabit fiyat aralığı. 0 girilirse sadece zone çizgisi çizilir.",
        )

    with settings_col2:
        momentum_threshold_pct = st.number_input(
            "Momentum threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=5.0,
            help="Momentum = abs(close - open) / mumun open fiyatının bulunduğu zone mesafesi. Varsayılan %35. Buffer hesaba dahil edilmez.",
        )

    with settings_col3:
        ichimoku_rr_multiplier = st.number_input(
            "1D Ichimoku TP multiplier (R)",
            min_value=0.1,
            max_value=10.0,
            value=1.7,
            step=0.1,
            help="Ichimoku TP hesabında SL mesafesinin kaç katının hedef alınacağını belirler. Örn. 2.0 = 2R.",
        )

with st.expander("Gösterge açıklamaları", expanded=False):
    st.markdown(
        f"""
- **EMA/SMA çizgileri:** Seçili hareketli ortalama çizgileri. Plotly legend'da sadece bunlar gösterilir.
- **Zone çizgisi:** Siyah kesikli yatay çizgi.
- **Zone buffer:** Zone çizgisinin altındaki/üstündeki hafif gri yatay bant.
- **Normal momentum:** Gri/yeşilimsi dikey gölge.
- **Counter momentum:** Kırmızı/pembe dikey gölge.
- **Momentum hesabı:** `abs(close - open)`, mumun **open** fiyatının bulunduğu iki manual zone arasındaki mesafeye göre ölçülür. Buffer hesaba dahil edilmez.
- **Resmî LONG/SHORT sinyali:** Eski yeşil/kırmızı üçgenler piyasa sinyal state değişimini gösterir.
- **Canlı pozisyon girişi:** Beyaz kenarlı büyük yeşil/kırmızı üçgen.
- **Canlı TP çizgisi:** Sarı basamaklı çizgi; Zone/SMA kaynağı hover içinde görünür.
- **TP gerçekleşmesi:** Sarı yıldız.
- **Pozisyon kapanışı:** Gri X.
- **Counter-momentum + ters sinyal geldiği halde RSI4 onayı olmadığı için korunan pozisyon:** Mor boş elmas.
- **15M sinyal referansı:** 15M'in kendi sinyali değildir; 1H sinyalinin 15M içi referans noktasıdır.
- **15M canlı 1H pozisyon ve TP:** Canlı strateji yine 1H kararıdır; 15M grafikte yalnızca aynı event/TP çizgileri daha yakın zaman kırılımında gösterilir.
- **1D Ichimoku TP/SL:** Sinyal mumunun kapanışı entry referansıdır. LONG için SL bulut alt sınırı, SHORT için SL bulut üst sınırıdır. TP = entry referansı ± {ichimoku_rr_multiplier:g} × SL mesafesi.
- **1D Ichimoku TP çizgisi:** Sarı düz yatay çizgi. Sinyal marker mumunda görünür ama TP çizgisi yalnızca bir sonraki günlük mum aynı Ichimoku sinyal state'iyle kapanırsa çizilir. Çizgi sinyal mumundan başlar; TP'ye temas edilirse orada, TP'den önce zıt Ichimoku sinyali gelirse zıt sinyalde biter.
        """
    )

snapshots = load_signal_snapshots(selected_pair)
hourly = load_candles(selected_pair, "1h")
m15 = load_candles(selected_pair, "15m")
daily = load_candles(selected_pair, "1d")
manual_zones = load_manual_zones(selected_pair)
counter_momentum_states = load_counter_momentum_states(selected_pair)
strategy_1h_state = load_strategy_1h_state(selected_pair)
strategy_1h_events = load_strategy_1h_events(selected_pair)
strategy_1h_tp_history = load_strategy_1h_tp_history(selected_pair)

if manual_zones.empty:
    st.info("Bu parite için aktif manual zone bulunamadı.")
else:
    st.caption(f"Aktif manual zone sayısı: {len(manual_zones)}")

if hourly.empty or m15.empty or daily.empty:
    st.warning("Bu coin için gerekli candle verileri Supabase'te yok. Önce pipeline'ı çalıştır.")
    st.stop()

# Dashboard grafiği açık 1H mumu da gösterebilir.
# Ancak resmi 1H sinyal markerları pipeline ile aynı şekilde sadece kapanmış 1H mumlardan üretilir.
hourly_closed_raw = get_closed_candles(hourly)

if hourly_closed_raw.empty:
    st.warning("1H için kapanmış mum bulunamadı. Resmi sinyal markerları üretilemez.")
    st.stop()

hourly = add_ema_rsi(hourly)
hourly_closed = add_ema_rsi(hourly_closed_raw)
m15 = add_ema_rsi(m15)
daily = add_ichimoku(daily)

hourly_chart = hourly.tail(48).copy()
m15_chart = m15.tail(96).copy()
#daily_chart = daily.tail(220).copy()

daily = add_ichimoku_signal_columns(
    daily,
    ichimoku_rr_multiplier=ichimoku_rr_multiplier,
)
daily_chart = daily.tail(220).copy()

hourly_signal_reference_events = all_signal_events(hourly_closed)
hourly_signal_markers = hourly_signal_reference_events[
    hourly_signal_reference_events["open_time"].isin(hourly_chart["open_time"])
].copy()

m15_intrabar_signal_reference_events = build_15m_intrabar_reference_markers(
    hourly_df=hourly_closed,
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

ichimoku_trade_cols = [
    "open_time",
    "ichimoku_entry_ref",
    "ichimoku_sl_level",
    "ichimoku_sl_distance",
    "ichimoku_tp_level",
    "ichimoku_rr",
    "ichimoku_tp_confirmed",
    "ichimoku_cloud_top",
    "ichimoku_cloud_bottom",
]
ichimoku_trade_cols = [col for col in ichimoku_trade_cols if col in daily.columns]

if not daily_signal_reference_events.empty:
    daily_signal_reference_events = daily_signal_reference_events.merge(
        daily[ichimoku_trade_cols],
        on="open_time",
        how="left",
    )

daily_signal_markers = daily_signal_reference_events[
    daily_signal_reference_events["open_time"].isin(daily_chart["open_time"])
].copy()

st.subheader("Live 1H strategy state")

if strategy_1h_state.empty:
    st.info("Live 1H strategy state kaydı henüz oluşmadı.")
else:
    live_state = strategy_1h_state.iloc[0]
    live_status = str(live_state.get("status") or "NA").upper()
    live_direction = str(live_state.get("direction") or "NA").upper()

    state_cols = st.columns(5)
    state_cols[0].metric("Status", live_status)
    state_cols[1].metric("Direction", live_direction)
    state_cols[2].metric("Entry", format(float(live_state["entry_price"]), ".2f") if pd.notna(live_state.get("entry_price")) else "NA")
    state_cols[3].metric("Target zone", format(float(live_state["target_zone"]), ".2f") if pd.notna(live_state.get("target_zone")) else "NA")
    state_cols[4].metric("TP", format(float(live_state["tp_trigger"]), ".2f") if pd.notna(live_state.get("tp_trigger")) else "NA")

    state_detail_cols = st.columns(4)
    state_detail_cols[0].caption(f"Entry reason: {live_state.get('entry_reason') or 'NA'}")
    state_detail_cols[1].caption(
        "Active zone: "
        + (
            f"{float(live_state['active_lower_zone']):.2f} – {float(live_state['active_upper_zone']):.2f}"
            if pd.notna(live_state.get("active_lower_zone")) and pd.notna(live_state.get("active_upper_zone"))
            else "NA"
        )
    )
    state_detail_cols[2].caption(f"TP source: {live_state.get('tp_source') or 'NA'}")
    state_detail_cols[3].caption(
        "Last processed closed candle: "
        + (_fmt_display_time(live_state.get("last_processed_closed_open_time")) or "NA")
    )

    if live_status == "PENDING_TP_DECISION":
        st.warning(
            "TP gerçekleşti; kapanan mum sonrası yeni pozisyon kararı bekleniyor. "
            f"Closed direction: {live_state.get('pending_closed_direction') or 'NA'} | "
            f"Exit: {float(live_state['pending_tp_exit_price']):.2f}"
            if pd.notna(live_state.get("pending_tp_exit_price"))
            else "TP gerçekleşti; kapanan mum sonrası yeni pozisyon kararı bekleniyor."
        )

with st.expander("Live 1H strategy event log", expanded=False):
    if strategy_1h_events.empty:
        st.info("Henüz canlı strateji olayı yok.")
    else:
        event_view = strategy_1h_events.sort_values("event_time", ascending=False).copy()
        event_view["event_time"] = event_view["event_time"].apply(_fmt_display_time)
        if "created_at" in event_view.columns:
            event_view["created_at"] = event_view["created_at"].apply(_fmt_display_time)
        if "details" in event_view.columns:
            event_view["details"] = event_view["details"].apply(_details_text)

        event_cols = [
            "event_time",
            "event_type",
            "trade_id",
            "direction",
            "price",
            "reason",
            "active_lower_zone",
            "active_upper_zone",
            "target_zone",
            "tp_source",
            "tp_raw_value",
            "tp_trigger",
            "details",
        ]
        event_cols = [c for c in event_cols if c in event_view.columns]
        st.dataframe(event_view[event_cols].head(100), use_container_width=True)

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
        "sma65",
        "sma120",
        "sma168",
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

col_1h_a, col_1h_b, col_1h_c, col_1h_d = st.columns(4)

with col_1h_a:
    show_zones_1h = st.toggle("1H yakın zone çizgileri", value=True, key="show_zones_1h")

with col_1h_b:
    show_signal_markers_1h = st.toggle("1H resmî sinyal markerları", value=False, key="show_signal_markers_1h_v2")

with col_1h_c:
    show_momentum_1h = st.toggle("1H momentum mumları", value=True, key="show_momentum_1h")

with col_1h_d:
    show_live_strategy_1h = st.toggle("1H canlı pozisyon ve TP", value=True, key="show_live_strategy_1h")

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
        counter_momentum_states=counter_momentum_states if show_momentum_1h else pd.DataFrame(),
        live_strategy_state=strategy_1h_state,
        live_strategy_events=strategy_1h_events,
        live_strategy_tp_history=strategy_1h_tp_history,
        show_live_strategy=show_live_strategy_1h,
    ),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)

st.plotly_chart(
    make_1h_rsi_chart(
        hourly_chart,
        f"{selected_pair} 1H — RSI4 / RSI14 / RSI52",
        zones=manual_zones,
        signal_markers=hourly_signal_markers,
        show_signal_markers=show_signal_markers_1h,
        show_momentum=show_momentum_1h,
        momentum_threshold_pct=momentum_threshold_pct,
        momentum_reference_events=hourly_signal_reference_events,
        live_strategy_events=strategy_1h_events,
        show_live_strategy=show_live_strategy_1h,
    ),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)

st.subheader(f"{selected_pair} — 15M (son 1 gün)")

col_15m_a, col_15m_b, col_15m_c, col_15m_d = st.columns(4)

with col_15m_a:
    show_zones_15m = st.toggle("15M yakın zone çizgileri", value=True, key="show_zones_15m")

with col_15m_b:
    show_signal_markers_15m = st.toggle("15M 1H sinyal referansı", value=False, key="show_signal_markers_15m_v2")

with col_15m_c:
    show_momentum_15m = st.toggle("15M momentum mumları", value=True, key="show_momentum_15m")

with col_15m_d:
    show_live_strategy_15m = st.toggle("15M canlı 1H pozisyon ve TP", value=True, key="show_live_strategy_15m")

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
        live_strategy_state=strategy_1h_state,
        live_strategy_events=strategy_1h_events,
        live_strategy_tp_history=strategy_1h_tp_history,
        show_live_strategy=show_live_strategy_15m,
    ),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)

st.subheader(f"{selected_pair} — 1D Ichimoku")

col_1d_a, col_1d_b, col_1d_c = st.columns(3)

with col_1d_a:
    show_zones_1d = st.toggle("1D yakın zone çizgileri", value=False, key="show_zones_1d")

with col_1d_b:
    show_signal_markers_1d = st.toggle("1D Ichimoku sinyal markerları", value=True, key="show_signal_markers_1d")

with col_1d_c:
    show_ichimoku_tp_lines = st.toggle("1D Ichimoku TP çizgileri", value=True, key="show_ichimoku_tp_lines")

st.plotly_chart(
    make_ichimoku_chart(
        daily_chart,
        f"{selected_pair} 1D — Ichimoku",
        zones=manual_zones,
        show_zones=show_zones_1d,
        zone_buffer=zone_buffer,
        signal_markers=daily_signal_markers,
        show_signal_markers=show_signal_markers_1d,
        tp_signal_events=daily_signal_reference_events,
        show_tp_lines=show_ichimoku_tp_lines,
    ),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)
