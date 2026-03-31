import os
import re
import io
import json
import base64
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from supabase import create_client

st.set_page_config(page_title="Crypto WMA Dashboard", layout="wide")

# =========================================================
# Constants
# =========================================================
LEVERAGE = 10.0            # 10x cross
NOON_TZ = "Europe/Istanbul"

# =========================================================
# Supabase creds
# =========================================================
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
        st.error("Supabase credentials not found.\nAdd in Secrets or env vars.")
        st.stop()
    return create_client(url, key)

supabase = supabase_client()

# =========================================================
# Data loaders
# =========================================================
@st.cache_data(ttl=300)
def load_snapshot():
    r = supabase.table("coin_wma_latest").select("*").execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ("close","wma_50","wma_200"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
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
        return pd.DataFrame(columns=["coin","date","decision"])
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df

@st.cache_data(ttl=300)
def load_price_history(days: int = 60):
    r = (supabase.table("coin_price_daily")
         .select("coin,date,close")
         .order("coin", desc=False)
         .order("date", desc=False)
         .execute())
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (df.sort_values(["coin","date"])
            .groupby("coin", as_index=False, group_keys=False)
            .tail(days))
    return df

@st.cache_data(ttl=300)
def load_wma_history(days: int = 60):
    r = (supabase.table("coin_wma")
         .select("coin,date,position")
         .order("coin", desc=False)
         .order("date", desc=False)
         .execute())
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = (df.sort_values(["coin","date"])
            .groupby("coin", as_index=False, group_keys=False)
            .tail(days))
    return df

# =========================================================
# Helpers
# =========================================================
def _ratio(decision: str) -> float:
    d = (decision or "").strip().lower()
    return 1.0 if d == "above both" else 0.5 if d == "between" else 0.0

def ratio_from_decision(d: str) -> float:
    d = (d or "").strip().lower()
    return 1.0 if d == "above both" else 0.5 if d == "between" else 0.0

def op_from_prev_to_now(prev: str, now: str, amt: float) -> str:
    rp, rn = ratio_from_decision(prev), ratio_from_decision(now)
    delta = rn - rp
    if abs(delta) < 1e-12:
        return "No action"
    action = "Buy" if delta > 0 else "Sell"
    qty = abs(delta) * max(amt, 0.0)
    return f"{action} {qty:.6f}"

# -------------------- Friend decisions upsert (JSON-safe) --------------------
def upsert_friend_decisions(rows: list[dict]) -> None:
    if not rows:
        return

    def _clean(r: dict) -> dict:
        out = {}
        out["coin"] = str(r.get("coin","")).upper().strip()
        d = r.get("date")
        if isinstance(d, pd.Timestamp):
            d = d.date()
        if hasattr(d, "isoformat"):
            out["date"] = d.isoformat()
        else:
            out["date"] = str(d)
        out["decision"] = str(r.get("decision","")).strip()
        return out

    for r in rows:
        supabase.table("friend_decisions").upsert(
            _clean(r), on_conflict="coin,date", returning="minimal"
        ).execute()

# =========================================================
# Sparkline renderer (hourly cache JSON)
# =========================================================
def render_spark_24h(series, noon_index, w50, w200) -> str | None:
    if not series or len(series) < 2:
        return None
    closes = np.array([float(p.get("c", np.nan)) for p in series], dtype=float)
    x = np.arange(len(closes))
    mask = ~np.isnan(closes)
    if mask.sum() < 2:
        return None

    a, b = np.polyfit(x[mask], closes[mask], 1)
    y_fit = a * x + b

    fig, ax = plt.subplots(figsize=(2.6, 0.6), dpi=150)
    ax.plot(x, closes, linewidth=1.0)
    ax.plot(x, y_fit, linestyle="--", linewidth=1.0)
    if noon_index is not None and 0 <= int(noon_index) < len(closes) and np.isfinite(closes[int(noon_index)]):
        ax.scatter([int(noon_index)], [closes[int(noon_index)]], s=14, zorder=5)
    if isinstance(w50, (int, float)) and np.isfinite(w50):
        ax.axhline(w50, color="green", linestyle=":", linewidth=1.0, label="WMA50")
    if isinstance(w200, (int, float)) and np.isfinite(w200):
        ax.axhline(w200, color="orange", linestyle="--", linewidth=1.0, label="WMA200")

    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    ymin, ymax = np.nanmin(closes[mask]), np.nanmax(closes[mask])
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def build_spark_df(df_latest: pd.DataFrame, df_cache: pd.DataFrame) -> pd.DataFrame:
    if df_cache.empty:
        return pd.DataFrame(columns=["coin","spark_24h"])
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
            try:
                series = json.loads(series)
            except Exception:
                series = []
        noon_idx = entry.get("noon_index")
        img = render_spark_24h(series, noon_idx, r.get("wma_50"), r.get("wma_200"))
        imgs.append({"coin": coin, "spark_24h": img})
    return pd.DataFrame(imgs)

# =========================================================
# Carry-forward aware analytics  (no merge_asof)
# =========================================================
def build_portfolio_curves(
    df_alloc,
    days: int = 60,
    leverage: float = 10.0,
    df_friend_extra: pd.DataFrame | None = None,
    override_last_n_days: int = 3,   # NEW: apply editor stance to last N chart dates
):
    """
    Returns:
      - model_curve: [date, value] indexed to 100 at start
      - friend_curve: [date, value] indexed to 100 at start
      - daily: [date, model, friend] USD notionals (10×)
      - detail: per-coin rows (date, coin, model_dec, friend_dec, etc.)
    Friend stance = latest saved decision as-of each date (carry-forward),
    optionally overridden by editor state for the last N chart dates.
    """
    df_close = load_price_history(days)
    df_pos   = load_wma_history(days)
    df_friend = load_friend_decisions()

    # --- normalize & (optional) inject today's snapshot like before ---
    # [keep your existing snapshot-injection code here if you added it]

    for dfx in (df_close, df_pos):
        if not dfx.empty:
            dfx["coin"] = dfx["coin"].astype(str).str.upper()
            dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce").dt.date

    if not df_friend.empty:
        df_friend = df_friend.copy()
        df_friend["coin"] = df_friend["coin"].astype(str).str.upper()
        df_friend["date"] = pd.to_datetime(df_friend["date"], errors="coerce").dt.date
    else:
        df_friend = pd.DataFrame(columns=["coin","date","decision"])

    # ---------- build the (coin,date) grid we will plot ----------
    df = (df_close.merge(df_pos, on=["coin","date"], how="inner")
                 .sort_values(["coin","date"]))
    if df.empty:
        empty_curve = pd.DataFrame(columns=["date","value"])
        empty_daily = pd.DataFrame(columns=["date","model","friend"])
        empty_detail = pd.DataFrame(columns=["date","coin","model_dec","friend_dec",
                                             "allocation","close","model_val","friend_val"])
        return empty_curve, empty_curve, empty_daily, empty_detail

    # ---------- APPLY EDITOR OVERRIDES TO LAST N DATES ----------
    if df_friend_extra is not None and not df_friend_extra.empty and override_last_n_days > 0:
        # last N unique dates in the chart grid
        last_dates = (
            pd.Series(pd.to_datetime(df["date"])).sort_values().dt.date.unique()[-override_last_n_days:]
        )
        extra = (df_friend_extra.copy()
                 .rename(columns={"friend_decision": "decision"}))
        extra["coin"] = extra["coin"].astype(str).str.upper()

        # cross-join coins × last_dates
        extra = extra.assign(key=1).merge(
            pd.DataFrame({"date": last_dates, "key": 1}), on="key"
        ).drop(columns="key")

        # append and keep the latest per (coin,date)
        df_friend = pd.concat([df_friend, extra[["coin","date","decision"]]], ignore_index=True)
        df_friend = (df_friend.sort_values(["coin","date"])
                              .drop_duplicates(["coin","date"], keep="last"))

    # ---------- carry-forward friend decision across the grid ----------
    if df_friend.empty:
        df["friend_decision_eff"] = df["position"]
    else:
        df_dates = df[["coin", "date"]].drop_duplicates().assign(_is_model_date=True)
        df_friend_marked = df_friend[["coin", "date", "decision"]].assign(_is_model_date=False)
        timeline = (
            pd.concat([df_friend_marked, df_dates.assign(decision=np.nan)], ignore_index=True)
              .sort_values(["coin", "date"], kind="mergesort")
        )
        timeline["decision_cf"] = (
            timeline.groupby("coin", as_index=False, group_keys=False)["decision"].ffill()
        )
        ff = timeline[timeline["_is_model_date"]][["coin","date","decision_cf"]]
        df = df.merge(ff, on=["coin","date"], how="left")
        df["friend_decision_eff"] = df["decision_cf"].fillna(df["position"])

    # ---------- compute per-coin and daily totals (unchanged) ----------
    alloc_map = {str(r["coin"]).upper(): float(r["allocation_amount"] or 0.0)
                 for _, r in (df_alloc.fillna(0)).iterrows()}

    rows = []
    for _, r in df.iterrows():
        coin = str(r["coin"]).upper()
        amt  = float(alloc_map.get(coin, 0.0))
        if amt <= 0: continue
        close = float(r["close"]) if pd.notnull(r["close"]) else np.nan
        if not np.isfinite(close): continue

        model_dec  = str(r["position"])
        friend_dec = str(r["friend_decision_eff"])
        model_val  = close * amt * _ratio(model_dec)  * leverage
        friend_val = close * amt * _ratio(friend_dec) * leverage

        rows.append({"date": r["date"], "coin": coin, "close": close, "allocation": amt,
                     "model_dec": model_dec, "friend_dec": friend_dec,
                     "model_val": model_val, "friend_val": friend_val})

    if not rows:
        empty_curve = pd.DataFrame(columns=["date","value"])
        empty_daily = pd.DataFrame(columns=["date","model","friend"])
        empty_detail = pd.DataFrame(columns=["date","coin","model_dec","friend_dec",
                                             "allocation","close","model_val","friend_val"])
        return empty_curve, empty_curve, empty_daily, empty_detail

    detail = pd.DataFrame(rows).sort_values(["date","coin"])
    daily = (detail.groupby("date", as_index=False)[["model_val","friend_val"]]
                  .sum()
                  .rename(columns={"model_val":"model", "friend_val":"friend"})
                  .sort_values("date"))

    def _normalize(col):
        s = daily[col].astype(float)
        base = float(s.iloc[0]) if len(s) else 0.0
        if base == 0: base = 1.0
        return (s / base) * 100.0

    model_curve  = pd.DataFrame({"date": daily["date"], "value": _normalize("model")})
    friend_curve = pd.DataFrame({"date": daily["date"], "value": _normalize("friend")})
    return model_curve, friend_curve, daily, detail



def compute_decision_match_rate_alltime(df_alloc):
    """
    Compare friend's carried-forward stance vs model_position across all dates,
    but only for coins with allocation_amount > 0.
    Returns (rate_pct, agree_count, total_compared).
    """
    df_pos = load_wma_history(days=9999)        # coin, date, position (model)
    df_friend_all = load_friend_decisions()     # coin, date, decision (friend saves)

    alloc_pos = df_alloc.copy()
    alloc_pos["coin"] = alloc_pos["coin"].astype(str).str.upper()
    alloc_set = set(alloc_pos.loc[alloc_pos["allocation_amount"] > 0, "coin"])

    if df_pos.empty or not alloc_set:
        return np.nan, 0, 0

    df_pos = df_pos.copy()
    df_pos["coin"] = df_pos["coin"].astype(str).str.upper()
    df_pos["date"] = pd.to_datetime(df_pos["date"], errors="coerce").dt.date
    df_pos = df_pos[df_pos["coin"].isin(alloc_set)]

    if df_friend_all.empty:
        return np.nan, 0, 0

    df_friend_all = df_friend_all.copy()
    df_friend_all["coin"] = df_friend_all["coin"].astype(str).str.upper()
    df_friend_all["date"] = pd.to_datetime(df_friend_all["date"], errors="coerce").dt.date

    # Build combined timeline and forward-fill per coin
    df_dates = df_pos[["coin", "date"]].dropna().drop_duplicates().assign(_is_model_date=True)
    df_friend_marked = df_friend_all[["coin", "date", "decision"]].assign(_is_model_date=False)

    timeline = (
        pd.concat([df_friend_marked, df_dates.assign(decision=np.nan)], ignore_index=True)
          .sort_values(["coin", "date"], kind="mergesort")
    )
    timeline["decision_cf"] = (
        timeline.sort_values(["coin", "date"])
                .groupby("coin", as_index=False, group_keys=False)["decision"]
                .ffill()
    )

    ff = timeline[timeline["_is_model_date"]][["coin", "date", "decision_cf"]]
    comp = df_pos.merge(ff, on=["coin", "date"], how="left")
    comp = comp[comp["decision_cf"].notna()].copy()

    total = len(comp)
    if total == 0:
        return np.nan, 0, 0

    comp["agree"] = (comp["position"].astype(str) == comp["decision_cf"].astype(str))
    agree = int(comp["agree"].sum())
    rate = 100.0 * agree / total
    return rate, agree, total

# =========================================================
# Page
# =========================================================
st.title("📈 Crypto WMA Dashboard")

df_latest = load_snapshot()
if df_latest.empty:
    st.info("No snapshot rows yet. Run the pipeline, then refresh.")
    st.stop()

df_alloc = load_allocations()
df_cache = load_intraday_cache()
df_friend = load_friend_decisions()

# Header metrics
last_dt = df_latest["date"].max()
coins_shown = df_latest["coin"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Coins shown", f"{coins_shown}")
c2.metric("Last updated (UTC)", last_dt.strftime("%Y-%m-%d") if pd.notnull(last_dt) else "—")
c3.metric("Leverage (x)", f"{LEVERAGE:g}")

# Local "today" in Istanbul (date only)
today_trt = datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=NOON_TZ).tz).date()

# ---------- Carry-forward prefill for today's editor ----------
df_today_friend = (
    df_friend[df_friend["date"] == today_trt][["coin", "decision"]]
      .rename(columns={"decision": "friend_today"})
)

df_prev_friend = (
    df_friend[df_friend["date"] < today_trt]
      .sort_values(["coin", "date"])
      .groupby("coin", as_index=False, group_keys=False)
      .tail(1)[["coin", "decision"]]
      .rename(columns={"decision": "friend_yesterday"})
)

spark_df = build_spark_df(df_latest, df_cache)

df = (df_latest
      .merge(df_alloc, on="coin", how="left")
      .merge(df_today_friend, on="coin", how="left")
      .merge(df_prev_friend, on="coin", how="left")
      .merge(spark_df, on="coin", how="left"))

df["allocation_amount"] = pd.to_numeric(df["allocation_amount"], errors="coerce").fillna(0.0)
df["close"] = pd.to_numeric(df["close"], errors="coerce")

# Prefill order: today's save → yesterday carry-forward → model position
df["friend_decision"] = (
    df["friend_today"]
      .fillna(df["friend_yesterday"])
      .fillna(df["position"])
)

# Sizes & USD values (today)
df["model_ratio"]  = df["position"].apply(ratio_from_decision)
df["friend_ratio"] = df["friend_decision"].apply(ratio_from_decision)
df["model_size"]   = df["allocation_amount"] * df["model_ratio"]
df["friend_size"]  = df["allocation_amount"] * df["friend_ratio"]
df["model_value_usd"]  = df["model_size"]  * df["close"] * LEVERAGE
df["friend_value_usd"] = df["friend_size"] * df["close"] * LEVERAGE

# Model operation delta from previous_position
df["model_op"] = df.apply(
    lambda r: op_from_prev_to_now(
        r.get("previous_position","Between"),
        r.get("position","Between"),
        float(r.get("allocation_amount",0.0))
    ),
    axis=1
)

# -------------------- Data editor --------------------
st.subheader("Latest snapshot (inline friend decisions)")
view_cols = [
    "coin","date","close","wma_50","wma_200","position","previous_position",
    "model_op","allocation_amount","model_size","friend_decision","friend_size",
    "model_value_usd","friend_value_usd","spark_24h"
]

colcfg = {
    "date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
    "close": st.column_config.NumberColumn(format="%.6f"),
    "wma_50": st.column_config.NumberColumn(format="%.6f"),
    "wma_200": st.column_config.NumberColumn(format="%.6f"),
    "allocation_amount": st.column_config.NumberColumn(format="%.6f",
        help="Full coin amount when Above; Between=half; Below=0"),
    "model_size": st.column_config.NumberColumn(format="%.6f"),
    "friend_size": st.column_config.NumberColumn(format="%.6f"),
    "model_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_value_usd": st.column_config.NumberColumn(format="%.2f"),
    "friend_decision": st.column_config.SelectboxColumn(
        "friend_decision",
        options=["Above both","Between","Below both"],
        help=f"Today's friend decision ({today_trt.isoformat()})"
    ),
    "spark_24h": st.column_config.ImageColumn("spark_24h", width="small",
        help="Hourly last 24h + trendline + noon marker + horizontal daily WMAs"),
}

disabled = [c for c in view_cols if c != "friend_decision"]

edited = st.data_editor(
    df[view_cols].sort_values("coin").reset_index(drop=True),
    use_container_width=True,
    column_config=colcfg,
    disabled=disabled,
    key="snapshot_editor",
)

if st.button("💾 Save friend decisions (today)"):
    before = df.set_index("coin")["friend_decision"]
    after  = pd.DataFrame(edited).set_index("coin")["friend_decision"]
    before, after = before.align(after, join="outer")
    changed = after[after != before].dropna()
    payload = [{"coin": c, "date": today_trt, "decision": v} for c, v in changed.items()]
    upsert_friend_decisions(payload)
    load_friend_decisions.clear()
    st.success(f"Saved {len(payload)} change(s). Reloading data…")
    st.rerun()

# -------------------- Portfolio snapshot metrics (today) --------------------
st.subheader("Portfolio snapshot — current notional (USD, 10×)")
model_total = float(df["model_value_usd"].sum())
friend_total = float(df["friend_value_usd"].sum())
cA, cB, cC, cD = st.columns(4)
cA.metric("Model total (USD)", f"{model_total:,.2f}")
cB.metric("Friend total (USD)", f"{friend_total:,.2f}")
cC.metric("Difference (USD)", f"{(model_total - friend_total):,.2f}")

# All-time decision match (carry-forward aware)
rate_all, agree_n, total_n = compute_decision_match_rate_alltime(df_alloc)
if np.isnan(rate_all):
    cD.metric("Decision match (all-time)", "—", help="No saved friend decisions yet.")
else:
    cD.metric("Decision match (all-time)", f"{rate_all:.1f}%", f"{agree_n}/{total_n}")

# -------------------- Comparison chart (toggle Indexed / USD) --------------------
st.subheader("Model vs Friend — performance (last 60 days)")

# Build df_friend_extra from current editor state so chart reflects unsaved edits
try:
    friend_extra = pd.DataFrame(edited)[["coin", "friend_decision"]]
except Exception:
    friend_extra = df[["coin", "friend_decision"]]

model_curve, friend_curve, daily, detail = build_portfolio_curves(
    df_alloc, days=60, leverage=LEVERAGE, df_friend_extra=friend_extra, override_last_n_days=3
)



if model_curve.empty or friend_curve.empty:
    st.info("Not enough history yet to draw the comparison chart.")
else:
    import matplotlib.dates as mdates

    mode = st.radio(
        "Chart scale",
        ["Indexed (start = 100)", "USD notional"],
        horizontal=True,
        index=0,
    )

    if mode == "Indexed (start = 100)":
        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=150)
        x = pd.to_datetime(model_curve["date"])
        ax.plot(x, model_curve["value"], label="Model", linewidth=1.6,
                linestyle="--", alpha=0.8, zorder=1)
        ax.plot(x, friend_curve["value"], label="Friend", linewidth=1.8,
                linestyle="-", alpha=1.0, zorder=2)
        ax.set_ylabel("Index (start = 100)")
    else:
        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=150)
        x = pd.to_datetime(daily["date"])
        ax.plot(x, daily["model"], label="Model", linewidth=1.6,
                linestyle="--", alpha=0.8, zorder=1)
        ax.plot(x, daily["friend"], label="Friend", linewidth=1.8,
                linestyle="-", alpha=1.0, zorder=2)
        ax.set_ylabel("USD notional (10×)")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)
    st.pyplot(fig, clear_figure=True)

# --- Diagnostics: show where the lines should differ on the last 7 chart dates
if not detail.empty:
    last_dates = (pd.to_datetime(detail["date"]).sort_values().unique())[-7:]
    diff_rows = detail[
        (pd.to_datetime(detail["date"]).isin(last_dates)) &
        (detail["allocation"] > 0) &
        (detail["model_dec"].astype(str) != detail["friend_dec"].astype(str))
    ].copy()

    st.caption("Differences used for the chart (last 7 dates, non-zero allocation):")
    if diff_rows.empty:
        st.write("No per-coin stance differences detected on the plotted dates.")
    else:
        diff_rows = diff_rows.sort_values(["date","coin"])
        diff_rows["date"] = pd.to_datetime(diff_rows["date"]).dt.date
        diff_rows["Δ_notional"] = (diff_rows["friend_val"] - diff_rows["model_val"]).round(2)
        st.dataframe(
            diff_rows[["date","coin","allocation","model_dec","friend_dec","close","Δ_notional"]],
            use_container_width=True
        )

