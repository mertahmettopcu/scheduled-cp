import os
import re
import io
import json
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

NOON_TZ = "Europe/Istanbul"
LEVERAGE = 10.0      # futures cross leverage
FEE_BPS = 4.0        # taker 0.04% per side
SLIP_BPS = 5.0       # assume small slippage per side
FUND_BPS = 0.0       # start with 0; set to 1.0 for ~1bp/day if desired

# -------------------- Supabase creds --------------------
def _read_supabase_creds():
    url = None; key = None
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
        st.error("Missing Supabase credentials.")
        st.stop()
    return create_client(url, key)

supabase = supabase_client()

# -------------------- Loads --------------------
@st.cache_data(ttl=300)
def load_snapshot():
    r = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ("close","wma_50","wma_200"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_coin_price_daily():
    # daily closes for P&L
    page_size = 2000; offset = 0; rows = []
    while True:
        resp = (supabase.table("coin_price_daily")
                .select("coin,date,close")
                .order("coin", desc=False).order("date", desc=False)
                .range(offset, offset + page_size - 1).execute())
        batch = resp.data or []
        if not batch: break
        rows.extend(batch); offset += page_size
        if len(batch) < page_size: break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_allocations():
    r = supabase.table("coin_allocations").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df["coin"] = df["coin"].astype(str)
        df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(ttl=300)
def load_intraday_cache():
    r = supabase.table("coin_hourly_24h_cache").select("*").execute()
    df = pd.DataFrame(r.data or [])
    return df

@st.cache_data(ttl=300)
def load_friend_decisions():
    r = supabase.table("friend_decisions").select("*").execute()
    data = r.data or []
    if not data:
        # Ensure required columns exist even when empty
        return pd.DataFrame(columns=["coin", "date", "decision"])
    df = pd.DataFrame(data)
    # normalize types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


# -------------------- Sparkline renderer (hourly cache) --------------------
def render_spark_24h(series, noon_index, w50, w200) -> str | None:
    """
    series: list of {"t": iso, "c": float}
    Returns base64 PNG for tiny ImageColumn.
    """
    if not series or len(series) < 2:
        return None
    # parse
    closes = np.array([float(p.get("c", np.nan)) for p in series], dtype=float)
    x = np.arange(len(closes))
    mask = ~np.isnan(closes)
    if mask.sum() < 2:
        return None
    # trend
    a, b = np.polyfit(x[mask], closes[mask], 1)
    y_fit = a*x + b

    fig, ax = plt.subplots(figsize=(2.6, 0.6), dpi=150)
    ax.plot(x, closes, linewidth=1.0)
    ax.plot(x, y_fit, linestyle="--", linewidth=1.0)
    if noon_index is not None and 0 <= noon_index < len(closes) and np.isfinite(closes[noon_index]):
        ax.scatter([noon_index], [closes[noon_index]], s=14, zorder=5)
    if isinstance(w50, (int,float)) and np.isfinite(w50):
        ax.axhline(w50, linestyle=":", linewidth=1.0)
    if isinstance(w200, (int,float)) and np.isfinite(w200):
        ax.axhline(w200, linestyle=":", linewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    ymin, ymax = np.nanmin(closes[mask]), np.nanmax(closes[mask])
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + (buf.getvalue()).encode("base64") if hasattr(bytes, "encode") else "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode("ascii")

# -------------------- Friend decision save --------------------
def upsert_friend_decisions(today_rows: list[dict]) -> None:
    if not today_rows:
        return
    # upsert one by one to keep it simple
    for r in today_rows:
        supabase.table("friend_decisions").upsert(
            r, on_conflict="coin,date", returning="minimal"
        ).execute()

# -------------------- P&L computation (coin amounts, 10x) --------------------
def map_decision_to_ratio(decision: str, long_only: bool = True) -> float:
    if not decision: return 0.0
    d = decision.strip().lower()
    if d == "above both":   return 1.0
    if d == "between":      return 0.5
    if d == "below both":   return 0.0 if long_only else -1.0  # long/short mode possible later
    return 0.0

def portfolio_curves(df_daily: pd.DataFrame,
                     df_latest: pd.DataFrame,
                     df_alloc: pd.DataFrame,
                     df_friends: pd.DataFrame,
                     long_only: bool = True):
    """
    Returns (equity_model, equity_friend) as DataFrames with columns [date, equity_index, value_usd]
    using coin allocation_amounts and 10x leverage, with fee/slip/funding.
    """
    if df_daily.empty or df_alloc.empty:
        return pd.DataFrame(), pd.DataFrame()

    # normalize
    px = df_daily.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.date
    px = px.dropna(subset=["date","close"])
    alloc = {r["coin"]: float(r["allocation_amount"]) for r in df_alloc.to_dict(orient="records")}

    # latest positions from snapshot (model)
    latest_positions = df_latest[["coin","position"]].dropna()
    pos_map = {r["coin"]: r["position"] for r in latest_positions.to_dict(orient="records")}

    # friend map by (coin,date)
    friend_map = {(r["coin"], r["date"]): r["decision"] for r in df_friends.to_dict(orient="records")}

    # build a per-coin time series of returns
    px = px.sort_values(["coin","date"])
    px["ret"] = px.groupby("coin")["close"].pct_change()

    # we need yesterday's decision to apply into today's return (avoid look-ahead)
    dates = sorted(px["date"].unique())
    coins = sorted(px["coin"].unique())

    # state: yesterday side ratio (0, 0.5, 1.0) for model & friend
    y_model = {c: 0.0 for c in coins}
    y_friend = {c: 0.0 for c in coins}

    eq_m = []  # per day model pnl sum
    eq_f = []
    eqidx_m, eqidx_f = 100.0, 100.0
    val_m, val_f = 0.0, 0.0  # absolute USD value if you want a starting bankroll; here we compute incremental

    # cost in fraction applied on change days (entry+exit)
    cost_turn = 2.0 * (FEE_BPS + SLIP_BPS) / 10000.0

    for d in dates:
        day = px[px["date"] == d]
        pnl_m = 0.0
        pnl_f = 0.0
        # build today's model/friend *decisions* (to be used tomorrow)
        model_dec_today = {}
        friend_dec_today = {}

        # model uses snapshot *latest* position as today's decision (approximation)
        # (If you want historical backfill, you can join historical positions table later.)
        for coin in day["coin"].unique():
            model_dec_today[coin] = pos_map.get(coin, "Between")

        for coin in day["coin"].unique():
            friend_dec_today[coin] = friend_map.get((coin, d), model_dec_today.get(coin, "Between"))

        # apply yesterday sides to today's returns
        for _, row in day.iterrows():
            coin = row["coin"]; r = row["ret"]
            if pd.isna(r):  # first day per coin
                continue
            amt = alloc.get(coin, 0.0)
            if amt <= 0:   # no allocation means no exposure
                continue
            px_prev = float(day.loc[day["coin"]==coin, "close"].shift(1).dropna().iloc[0]) if True else None

            # model side & pnl
            side_m = y_model.get(coin, 0.0)
            exposure_m = amt * side_m * LEVERAGE  # coin units * leverage
            pnl_m += exposure_m * r * px_prev if px_prev else 0.0

            # friend side & pnl
            side_f = y_friend.get(coin, 0.0)
            exposure_f = amt * side_f * LEVERAGE
            pnl_f += exposure_f * r * px_prev if px_prev else 0.0

            # funding cost (optional) when in position
            if abs(side_m) > 0:
                pnl_m -= amt * abs(side_m) * LEVERAGE * (FUND_BPS/10000.0) * (px_prev if px_prev else 0.0)
            if abs(side_f) > 0:
                pnl_f -= amt * abs(side_f) * LEVERAGE * (FUND_BPS/10000.0) * (px_prev if px_prev else 0.0)

        # update equity index: scale pnl by notional 100 baseline via simple normalization
        # For index scaling, divide daily PnL by a normalizer (sum of full notional across coins).
        # We use sum(amt * LEVERAGE * yesterday_price) to approximate. Simpler: convert pnl -> daily return relative to 100.
        # For clarity here, we map pnl→daily_ret via a fixed normalizer: total_full_usd = sum(amt * last_price).
        total_full_usd = 0.0
        for coin in day["coin"].unique():
            sub = day[day["coin"]==coin]
            px_prev = float(sub["close"].shift(1).dropna().iloc[0]) if sub["close"].shift(1).notna().any() else None
            if px_prev:
                total_full_usd += alloc.get(coin,0.0) * px_prev * LEVERAGE

        if total_full_usd > 0:
            ret_m = pnl_m / total_full_usd
            ret_f = pnl_f / total_full_usd
        else:
            ret_m = ret_f = 0.0

        eqidx_m *= (1.0 + ret_m)
        eqidx_f *= (1.0 + ret_f)

        eq_m.append({"date": d, "equity_index": eqidx_m})
        eq_f.append({"date": d, "equity_index": eqidx_f})

        # now set today's side (used tomorrow)
        for coin in day["coin"].unique():
            y_model[coin] = map_decision_to_ratio(model_dec_today.get(coin, "Between"), long_only=True)
            y_friend[coin] = map_decision_to_ratio(friend_dec_today.get(coin, "Between"), long_only=True)

    eqM = pd.DataFrame(eq_m); eqF = pd.DataFrame(eq_f)
    # Convert to USD using the *current* notional at last date (index-to-USD)
    if not eqM.empty:
        last_d = eqM["date"].max()
        last_day = px[px["date"] == last_d]
        total_usd = 0.0
        for _, r in last_day.iterrows():
            total_usd += alloc.get(r["coin"],0.0) * float(r["close"]) * LEVERAGE
        eqM["value_usd"] = (eqM["equity_index"] / 100.0) * total_usd if total_usd>0 else np.nan
        eqF["value_usd"] = (eqF["equity_index"] / 100.0) * total_usd if total_usd>0 else np.nan
    return eqM, eqF

# ==================== PAGE ====================
st.title("📈 Crypto WMA Dashboard")

df_latest = load_snapshot()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh."); st.stop()

df_daily  = load_coin_price_daily()
df_alloc  = load_allocations()
df_cache  = load_intraday_cache()
df_friend = load_friend_decisions()

# Header
last_dt = df_latest["date"].max()
coins_shown = df_latest["coin"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Coins shown", f"{coins_shown}")
c2.metric("Last updated (UTC)", last_dt.strftime("%Y-%m-%d") if pd.notnull(last_dt) else "—")
c3.metric("Leverage (x)", f"{LEVERAGE:g}")

# Friend decisions input for TODAY
today_trt = datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=NOON_TZ).tz).date()
st.subheader(f"Friend decisions for {today_trt.isoformat()}")

options = ["Above both","Between","Below both"]
edit_rows = []
cols = st.columns(4)
with cols[0]: st.write("**coin**")
with cols[1]: st.write("**model position**")
with cols[2]: st.write("**friend_decision (today)**")
with cols[3]: st.write("**allocation amount**")

for _, row in df_latest.sort_values("coin").iterrows():
    coin = row["coin"]
    model_pos = row.get("position","Between") or "Between"
    alloc_amt = float(df_alloc.loc[df_alloc["coin"]==coin, "allocation_amount"].max()) if coin in set(df_alloc["coin"]) else 0.0
    # prefill friend value if already saved today
    prev = pd.Series(dtype=object)
    if not df_friend.empty and {"coin","date","decision"}.issubset(df_friend.columns):
        prev = df_friend.loc[(df_friend["coin"]==coin) & (df_friend["date"]==today_trt), "decision"]
    pre = prev.iloc[0] if not prev.empty else model_pos

    c = st.columns(4)
    with c[0]: st.write(coin)
    with c[1]: st.write(model_pos)
    with c[2]:
        choice = st.selectbox(f"friend_{coin}", options, index=options.index(pre) if pre in options else 1, key=f"friend_{coin}")
    with c[3]: st.write(f"{alloc_amt:g}")
    edit_rows.append({"coin": coin, "date": today_trt, "decision": choice})

if st.button("💾 Save today's friend decisions"):
    upsert_friend_decisions(edit_rows)
    st.success("Saved. Reload to reflect in metrics/curves.")

# Build tiny sparkline images
def build_spark_df():
    imgs = []
    cache_map = {r["coin"]: r for r in df_cache.to_dict(orient="records")}
    for _, r in df_latest.iterrows():
        coin = r["coin"]
        entry = cache_map.get(coin)
        if not entry:
            imgs.append({"coin": coin, "spark_24h": None})
            continue
        series = entry.get("series", [])
        if isinstance(series, str):
            try: series = json.loads(series)
            except Exception: series = []
        noon_idx = entry.get("noon_index")
        img = render_spark_24h(series, noon_idx, r.get("wma_50"), r.get("wma_200"))
        imgs.append({"coin": coin, "spark_24h": img})
    return pd.DataFrame(imgs)

spark_df = build_spark_df()
df_show = df_latest.merge(df_alloc, on="coin", how="left").merge(spark_df, on="coin", how="left")
df_show["allocation_amount"] = df_show["allocation_amount"].fillna(0.0)

# status (unchanged)
def status_badge(row):
    if row.get("position") and row.get("previous_position") and row["position"] != row["previous_position"]:
        return "🔔 Change"
    return "✓ Stable"
df_show["status"] = df_show.apply(status_badge, axis=1)

st.subheader("Latest snapshot (with hourly sparkline)")
cols = ["coin","date","close","wma_50","wma_200","position","previous_position","status","allocation_amount","spark_24h"]
st.dataframe(
    df_show[cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config={
        "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "close": st.column_config.NumberColumn(format="%.6f"),
        "wma_50": st.column_config.NumberColumn(format="%.6f"),
        "wma_200": st.column_config.NumberColumn(format="%.6f"),
        "allocation_amount": st.column_config.NumberColumn(format="%.6f", help="Full coin amount if Above both; Between=half; Below=0"),
        "spark_24h": st.column_config.ImageColumn("spark_24h", width="small",
            help="Hourly last 24h with trendline, noon marker, horizontal daily WMAs"),
    },
)

# ---- Portfolio Model vs Friend (index + USD now) ----
st.subheader("Portfolio P&L — Model vs Friend")

# reload friend decisions to include any just-saved rows
df_friend = load_friend_decisions()
eqM, eqF = portfolio_curves(df_daily, df_latest, df_alloc, df_friend, long_only=True)

if eqM.empty:
    st.info("Not enough data to compute curves yet.")
else:
    # metrics
    latest_idx_M = float(eqM.iloc[-1]["equity_index"])
    latest_idx_F = float(eqF.iloc[-1]["equity_index"])
    latest_val_M = float(eqM.iloc[-1].get("value_usd", np.nan))
    latest_val_F = float(eqF.iloc[-1].get("value_usd", np.nan))
    colA, colB, colC = st.columns(3)
    with colA: st.metric("Model equity (index)", f"{latest_idx_M:.2f}")
    with colB: st.metric("Friend equity (index)", f"{latest_idx_F:.2f}")
    with colC:
        if np.isfinite(latest_val_M) and np.isfinite(latest_val_F):
            st.metric("Current value diff (USD)", f"{(latest_val_M-latest_val_F):,.2f}")
    # chart
    fig, ax = plt.subplots(figsize=(9,3), dpi=150)
    ax.plot(pd.to_datetime(eqM["date"]), eqM["equity_index"], label="Model", linewidth=1.2)
    ax.plot(pd.to_datetime(eqF["date"]), eqF["equity_index"], label="Friend", linewidth=1.2)
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (index=100)")
    ax.grid(True, alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig, clear_figure=True)
