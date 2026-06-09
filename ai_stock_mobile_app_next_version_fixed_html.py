import html
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Stock Mobile", page_icon="📈", layout="centered")

DEFAULT_WATCHLIST = [
    "VUG", "NVDA", "SMH", "VGT", "QQQM", "VOO", "ADBE",
    "AMD", "AVGO", "MSFT", "META", "GOOGL", "ETSY",
    "TSM", "MU", "ARM", "OXY", "MO", "HRL"
]


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def pct_distance(price, level):
    if level == 0 or pd.isna(level):
        return None
    return (price - level) / level * 100


@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, period: str = "1y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    if len(df) < 60:
        return None

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)
    df["ATR14_%"] = df["ATR14"] / df["Close"] * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - macd_signal
    df["Volume20Avg"] = df["Volume"].rolling(20).mean()

    daily_return = df["Close"].pct_change()
    df["Vol20_Annual_%"] = daily_return.rolling(20).std() * np.sqrt(252) * 100
    df["Vol60_Annual_%"] = daily_return.rolling(60).std() * np.sqrt(252) * 100

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Width_%"] = ((sma20 + 2 * std20) - (sma20 - 2 * std20)) / sma20 * 100

    return df.dropna()


@st.cache_data(ttl=900)
def load_volatility_data(ticker: str, interval: str):
    settings = {
        "5m": {"period": "5d", "scale": 78 * 252, "window": 20, "label": "5-minute"},
        "30m": {"period": "1mo", "scale": 13 * 252, "window": 20, "label": "30-minute"},
        "60m": {"period": "3mo", "scale": 6.5 * 252, "window": 20, "label": "hourly"},
        "1d": {"period": "1y", "scale": 252, "window": 20, "label": "daily"},
        "1wk": {"period": "5y", "scale": 52, "window": 20, "label": "weekly"},
    }
    s = settings[interval]
    df = yf.download(ticker, period=s["period"], interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return None, s
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df["Return"] = df["Close"].pct_change()
    df["Rolling_Vol_%"] = df["Return"].rolling(s["window"]).std() * np.sqrt(s["scale"]) * 100
    return df.dropna(), s


def links_for(ticker):
    symbol = quote(ticker.upper())
    return {
        "Yahoo Chart": f"https://finance.yahoo.com/chart/{symbol}",
        "Yahoo Quote": f"https://finance.yahoo.com/quote/{symbol}",
        "Fidelity": f"https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol={symbol}",
    }


def grade_from_score(score):
    if score >= 8:
        return "A"
    if score >= 6:
        return "B"
    if score >= 4:
        return "C"
    return "D"


def visual_status(category, score):
    if category == "Within Buy" and score >= 8:
        return "🟢", "High Conviction Buy"
    if category == "Within Buy":
        return "🟩", "Moderate Buy"
    if category == "Near Buy":
        return "🟩", "Near Buy"
    if category == "Active Watch":
        return "🟡", "Watch"
    if category == "Near Sell":
        return "🟠", "Caution"
    if category == "Within Sell" and score <= 3:
        return "🟥", "Hard Sell"
    if category == "Within Sell":
        return "🔴", "Sell"
    return "⚪", "Neutral"


def card_background(category, score):
    if category == "Within Buy" and score >= 8:
        return "#e8f8ee"
    if category in ["Within Buy", "Near Buy"]:
        return "#f0fbf3"
    if category == "Active Watch":
        return "#fffbea"
    if category == "Near Sell":
        return "#fff3e0"
    if category == "Within Sell" and score <= 3:
        return "#fde7e7"
    if category == "Within Sell":
        return "#fff0f0"
    return "white"


def volatility_percentile(df: pd.DataFrame) -> float:
    vol = df["Vol20_Annual_%"].dropna()
    if len(vol) < 30:
        return np.nan
    latest = vol.iloc[-1]
    return float((vol <= latest).mean() * 100)


def build_scores(df: pd.DataFrame):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"])
    rsi = float(latest["RSI14"])
    macd_hist = float(latest["MACD_Hist"])
    prev_macd_hist = float(previous["MACD_Hist"])
    atr = float(latest["ATR14"])
    atr_pct = float(latest["ATR14_%"])
    dist20 = pct_distance(price, ema20)
    vol_pctile = volatility_percentile(df)

    trend_score = 0
    if price > ema20:
        trend_score += 3
    if ema20 > ema50:
        trend_score += 3
    if ema50 > ema200:
        trend_score += 2
    if macd_hist > prev_macd_hist:
        trend_score += 2

    risk_score = 0
    if rsi > 70:
        risk_score += 3
    elif rsi > 65:
        risk_score += 2
    elif rsi < 35:
        risk_score += 2
    if dist20 is not None and dist20 > 8:
        risk_score += 3
    elif dist20 is not None and dist20 > 4:
        risk_score += 2
    if atr_pct > 4:
        risk_score += 2
    elif atr_pct > 2.5:
        risk_score += 1
    if not np.isnan(vol_pctile):
        if vol_pctile > 80:
            risk_score += 2
        elif vol_pctile > 60:
            risk_score += 1
    risk_score = min(risk_score, 10)

    timing_score = 5
    if price <= ema20:
        timing_score += 2
    if dist20 is not None and -3 <= dist20 <= 2:
        timing_score += 2
    elif dist20 is not None and dist20 > 6:
        timing_score -= 2
    if 40 <= rsi <= 60:
        timing_score += 1
    elif rsi > 70:
        timing_score -= 2
    elif rsi < 35:
        timing_score -= 1
    if macd_hist > prev_macd_hist:
        timing_score += 1
    if risk_score >= 8:
        timing_score -= 2
    timing_score = max(0, min(10, timing_score))

    return {
        "Trend_Score": round(trend_score, 1),
        "Risk_Score": round(risk_score, 1),
        "Timing_Score": round(timing_score, 1),
        "ATR": round(atr, 2),
        "ATR_%": round(atr_pct, 2),
        "Vol20_%": round(float(latest["Vol20_Annual_%"]), 2),
        "Vol60_%": round(float(latest["Vol60_Annual_%"]), 2),
        "Volatility_Percentile": round(vol_pctile, 1) if not np.isnan(vol_pctile) else None,
        "BB_Width_%": round(float(latest["BB_Width_%"]), 2),
        "EMA200": round(ema200, 2),
        "Aggressive_Buy": round(price - 0.5 * atr, 2),
        "Normal_Buy": round(price - 1.0 * atr, 2),
        "Strong_Buy": round(price - 1.5 * atr, 2),
    }


def screen_ticker(ticker: str):
    df = load_stock_data(ticker)
    if df is None or len(df) < 60:
        return None
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI14"])
    volume = float(latest["Volume"])
    volume_avg = float(latest["Volume20Avg"])
    macd_hist = float(latest["MACD_Hist"])
    prev_macd_hist = float(previous["MACD_Hist"])
    distance_to_ema20 = pct_distance(price, ema20)
    volume_ratio = volume / volume_avg if volume_avg else 0
    score = 0
    notes = []

    if price > ema20:
        score += 2
        notes.append("Price above 20 EMA")
    else:
        notes.append("Price below 20 EMA")
    if ema20 > ema50:
        score += 2
        notes.append("20 EMA above 50 EMA")
    else:
        notes.append("20 EMA below 50 EMA")
    if 45 <= rsi <= 65:
        score += 2
        notes.append("RSI healthy")
    elif rsi > 70:
        notes.append("RSI overbought")
    elif rsi < 40:
        notes.append("RSI weak")
    if macd_hist > prev_macd_hist:
        score += 2
        notes.append("MACD histogram improving")
    else:
        notes.append("MACD histogram weakening")
    if volume_ratio >= 1.2:
        score += 1
        notes.append("Volume above average")

    category = "Active Watch"
    if price > ema20 and ema20 > ema50 and 45 <= rsi <= 65 and macd_hist > prev_macd_hist and -1 <= distance_to_ema20 <= 3:
        category = "Within Buy"
    elif ema20 > ema50 and 40 <= rsi <= 68 and -3 <= distance_to_ema20 <= 5:
        category = "Near Buy"
    if price < ema20 and macd_hist < 0 and rsi < 45:
        category = "Within Sell"
    elif price > ema20 and distance_to_ema20 is not None and distance_to_ema20 >= 8 and (rsi >= 70 or macd_hist < prev_macd_hist):
        category = "Near Sell"

    emoji, visual_label = visual_status(category, score)
    result = {
        "Ticker": ticker,
        "Category": category,
        "Visual_Status": visual_label,
        "Emoji": emoji,
        "Price": round(price, 2),
        "Score": score,
        "Grade": grade_from_score(score),
        "RSI": round(rsi, 1),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "Distance_to_EMA20_%": round(distance_to_ema20, 2),
        "MACD_Hist": round(macd_hist, 3),
        "Volume_Ratio": round(volume_ratio, 2),
        "Notes": "; ".join(notes),
    }
    result.update(build_scores(df))
    result.update(links_for(ticker))
    return result


def risk_label(score):
    if score <= 3:
        return "🟢 Low risk"
    if score <= 6:
        return "🟡 Normal risk"
    if score <= 8:
        return "🟠 High risk"
    return "🔴 Very high risk"


def timing_label(score):
    if score >= 8:
        return "🟢 Better timing"
    if score >= 6:
        return "🟡 Watch / partial buy"
    if score >= 4:
        return "🟠 Wait for pullback"
    return "🔴 Poor timing"



def volatility_status_label(current_vol: float, avg_vol: float, percentile: float):
    if pd.isna(current_vol) or pd.isna(avg_vol) or pd.isna(percentile):
        return "⚪ Unknown Volatility", "Unknown"
    if percentile >= 80 or current_vol >= avg_vol * 1.5:
        return "🔴 High Volatility", "High"
    if percentile >= 60 or current_vol >= avg_vol * 1.15:
        return "🟠 Elevated Volatility", "Elevated"
    if percentile >= 30:
        return "🟡 Normal Volatility", "Normal"
    return "🟢 Low Volatility", "Low"


def score_phrase(score, kind):
    score = float(score)
    if kind == "timing":
        if score >= 8:
            return "🟢 Good"
        if score >= 6:
            return "🟡 Watch / partial"
        if score >= 4:
            return "🟠 Wait"
        return "🔴 Poor"
    if kind == "trend":
        if score >= 8:
            return "🟢 Strong Uptrend"
        if score >= 6:
            return "🟡 Uptrend"
        if score >= 4:
            return "🟠 Weak Trend"
        return "🔴 Downtrend"
    if kind == "risk":
        if score <= 3:
            return "🟢 Low"
        if score <= 5:
            return "🟢 Low-Normal"
        if score <= 7:
            return "🟡 Elevated"
        return "🔴 High"
    return "⚪ Neutral"


def rsi_phrase(rsi):
    rsi = float(rsi)
    if rsi < 30:
        return "🟢 Oversold"
    if rsi < 45:
        return "🟡 Weak"
    if rsi <= 65:
        return "🟢 Healthy"
    if rsi <= 70:
        return "🟠 Warm"
    return "🔴 Overbought"


def vol_phrase(vol20):
    vol20 = float(vol20)
    if vol20 < 25:
        return "🟢 Low"
    if vol20 < 40:
        return "🟡 Normal"
    if vol20 < 60:
        return "🟡 Elevated"
    return "🔴 High"


def ema_distance_phrase(dist):
    dist = float(dist)
    if dist > 8:
        return "🔴 Too extended"
    if dist > 3:
        return "🟠 Warm"
    if dist >= -3:
        return "🟢 Ideal"
    if dist >= -6:
        return "🟢 Pullback buy zone"
    return "🟠 Deep pullback"


def ai_verdict(row):
    timing = float(row["Timing_Score"])
    risk = float(row["Risk_Score"])
    trend = float(row["Trend_Score"])
    dist = float(row["Distance_to_EMA20_%"])
    if timing >= 8 and risk <= 5 and trend >= 7 and dist <= 3:
        return "🟢 GOOD BUY CANDIDATE", "Trend strong; risk controlled; not overheated; near EMA20"
    if timing >= 6 and trend >= 6 and risk <= 7:
        return "🟡 WATCH / PARTIAL BUY", "Setup is acceptable, but wait for a better pullback if possible"
    if risk >= 8 or dist > 8:
        return "🔴 WAIT", "Risk or extension is high; avoid chasing"
    return "🟠 NEUTRAL", "No strong edge right now"

def card(row, key_prefix="main"):
    ticker = html.escape(str(row["Ticker"]))
    category = html.escape(str(row["Category"]))
    visual_status_text = html.escape(str(row["Visual_Status"]))
    notes = html.escape(str(row["Notes"]))
    background = card_background(row["Category"], row["Score"])

    timing_text = score_phrase(row["Timing_Score"], "timing")
    trend_text = score_phrase(row["Trend_Score"], "trend")
    risk_text = score_phrase(row["Risk_Score"], "risk")
    rsi_text = rsi_phrase(row["RSI"])
    vol_text = vol_phrase(row["Vol20_%"])
    ema_text = ema_distance_phrase(row["Distance_to_EMA20_%"])
    daily_swing = float(row["ATR"])
    verdict, verdict_reason = ai_verdict(row)

    st.markdown(
        textwrap.dedent(f'''
        <div style="border:1px solid #d9d9d9;border-radius:18px;padding:14px;margin-bottom:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);background-color:{background};">
            <h3 style="margin:0 0 8px 0;">{row['Emoji']} {ticker} — {visual_status_text}</h3>
            <p style="margin:6px 0;"><b>Setup:</b> {category}</p>
            <p style="margin:6px 0;"><b>Price:</b> ${row['Price']} &nbsp; <b>Score:</b> {row['Score']}/9 &nbsp; <b>Grade:</b> {row['Grade']}</p>

            <div style="margin-top:10px;padding:10px;border-radius:12px;background-color:rgba(255,255,255,0.72);">
                <p style="margin:5px 0;"><b>{timing_text} Buy Timing:</b> {row['Timing_Score']}/10</p>
                <p style="margin:5px 0;"><b>{trend_text} Trend:</b> {row['Trend_Score']}/10</p>
                <p style="margin:5px 0;"><b>{risk_text} Risk:</b> {row['Risk_Score']}/10</p>
                <p style="margin:5px 0;"><b>{rsi_text} RSI:</b> {row['RSI']}</p>
                <p style="margin:5px 0;"><b>🟡 Daily Swing:</b> ±${daily_swing:.2f} (ATR {row['ATR_%']}%)</p>
                <p style="margin:5px 0;"><b>{vol_text} Volatility:</b> {row['Vol20_%']}%</p>
                <p style="margin:5px 0;"><b>{ema_text} Distance to EMA20:</b> {row['Distance_to_EMA20_%']}%</p>
            </div>

            <div style="margin-top:10px;padding:10px;border-radius:12px;background-color:rgba(255,255,255,0.72);">
                <p style="margin:5px 0;font-size:18px;"><b>AI Verdict: {verdict}</b></p>
                <p style="margin:5px 0;">{html.escape(verdict_reason)}</p>
            </div>

            <p style="margin:8px 0;"><b>Notes:</b> {notes}</p>
            <p style="margin:8px 0 0 0;">
                <a href="{row['Yahoo Chart']}" target="_blank">Yahoo Chart</a>
                &nbsp;|&nbsp;
                <a href="{row['Yahoo Quote']}" target="_blank">Yahoo Quote</a>
                &nbsp;|&nbsp;
                <a href="{row['Fidelity']}" target="_blank">Fidelity</a>
                &nbsp;|&nbsp;
                <b>Open dashboards below</b>
            </p>
        </div>
        '''),
        unsafe_allow_html=True,
    )

    with st.expander(f"📊 {ticker} Dashboards: Timing | Risk | Volatility", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Buy Timing", "Risk", "Volatility Chart"])
        with tab1:
            st.subheader(f"{ticker} Buy Timing")
            verdict, verdict_reason = ai_verdict(row)
            st.markdown(f"### {verdict}")
            st.caption(verdict_reason)
            c1, c2 = st.columns(2)
            c1.metric("Timing Score", f"{row['Timing_Score']}/10", timing_label(row["Timing_Score"]))
            c2.metric("Distance to EMA20", f"{row['Distance_to_EMA20_%']}%", ema_distance_phrase(row["Distance_to_EMA20_%"]))
            st.markdown(f'''
**Suggested buy levels based on ATR**

- Aggressive buy: **${row['Aggressive_Buy']}**
- Normal buy: **${row['Normal_Buy']}**
- Strong buy: **${row['Strong_Buy']}**

Best intraday windows to check: **10:30–11:30 ET** and **2:00–3:30 ET**.
''')
        with tab2:
            st.subheader(f"{ticker} Risk Dashboard")
            c1, c2 = st.columns(2)
            c1.metric("Risk Score", f"{row['Risk_Score']}/10", risk_label(row["Risk_Score"]))
            c2.metric("Daily Swing", f"±${row['ATR']}", f"ATR {row['ATR_%']}%")
            c3, c4 = st.columns(2)
            c3.metric("RSI", row["RSI"], rsi_phrase(row["RSI"]))
            c4.metric("BB Width", f"{row['BB_Width_%']}%")
            st.metric("Volatility Percentile", "N/A" if row["Volatility_Percentile"] is None else f"{row['Volatility_Percentile']}%")
        with tab3:
            st.subheader(f"{ticker} Volatility Chart")
            interval_label = st.radio("Chart interval", ["5m", "30m", "60m", "1d", "1wk"], horizontal=True, key=f"interval_{key_prefix}_{ticker}")
            vol_df, settings = load_volatility_data(str(row["Ticker"]), interval_label)
            if vol_df is None or vol_df.empty:
                st.warning("No volatility data available for this interval.")
            else:
                latest_vol = float(vol_df["Rolling_Vol_%"].iloc[-1])
                avg_30 = float(vol_df["Rolling_Vol_%"].tail(30).mean())
                percentile = float((vol_df["Rolling_Vol_%"] <= latest_vol).mean() * 100)
                status_text, risk_level = volatility_status_label(latest_vol, avg_30, percentile)

                st.markdown("### Volatility Status")
                st.markdown(f"## {status_text}")
                c1, c2 = st.columns(2)
                c1.metric("Current", f"{latest_vol:.2f}%")
                c2.metric("30-Day Average", f"{avg_30:.2f}%")
                c3, c4 = st.columns(2)
                c3.metric("Percentile", f"{percentile:.0f}%")
                c4.metric("Risk Level", risk_level)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vol_df.index, y=vol_df["Rolling_Vol_%"], mode="lines", name="Rolling volatility %"))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Annualized volatility %", xaxis_title="Time")
                st.plotly_chart(fig, use_container_width=True, key=f"vol_chart_{key_prefix}_{ticker}_{interval_label}")


st.title("📈 AI Stock Mobile")
st.caption("Mobile-friendly AI stock/ETF setup scanner, buy timing, risk, and volatility chart")
st.warning("Educational/research tool only. Not financial advice. Confirm with Yahoo/Fidelity before trading.")

with st.expander("Color Legend", expanded=False):
    st.markdown("""
    - 🟢 **High Conviction Buy**: Within Buy + Score 8–9
    - 🟩 **Moderate/Near Buy**: Within Buy below 8, or Near Buy
    - 🟡 **Watch**: Active Watch
    - 🟠 **Caution**: Near Sell
    - 🔴 **Sell**: Within Sell
    - 🟥 **Hard Sell**: Within Sell + Score 0–3
    """)

with st.expander("Watchlist Settings", expanded=False):
    watchlist_text = st.text_area("Tickers separated by commas", value=", ".join(DEFAULT_WATCHLIST), height=120)

scan_button = st.button("🔍 Run Scan", use_container_width=True)
watchlist = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]

if scan_button or "mobile_results" not in st.session_state:
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(watchlist):
        result = screen_ticker(ticker)
        if result:
            results.append(result)
        progress.progress((i + 1) / max(len(watchlist), 1))
    st.session_state["mobile_results"] = pd.DataFrame(results)
    st.session_state["mobile_last_scan"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M:%S %p %Z")

df = st.session_state.get("mobile_results", pd.DataFrame())

if df.empty:
    st.info("No data loaded yet. Tap Run Scan.")
else:
    st.caption(f"Last scan: {st.session_state.get('mobile_last_scan', 'N/A')}")
    metric1, metric2 = st.columns(2)
    metric1.metric("High Buy", len(df[df["Visual_Status"] == "High Conviction Buy"]))
    metric2.metric("Hard Sell", len(df[df["Visual_Status"] == "Hard Sell"]))
    metric3, metric4 = st.columns(2)
    metric3.metric("Buy/Near Buy", len(df[df["Category"].isin(["Within Buy", "Near Buy"])]))
    metric4.metric("Sell Risk", len(df[df["Category"].isin(["Within Sell", "Near Sell"])]))

    tabs = st.tabs(["Buy", "Watch", "Sell", "All"])
    with tabs[0]:
        buy_df = df[df["Category"].isin(["Within Buy", "Near Buy"])].sort_values("Timing_Score", ascending=False)
        if buy_df.empty:
            st.info("No buy setups now.")
        else:
            for _, row in buy_df.iterrows():
                card(row, key_prefix="buy")
    with tabs[1]:
        watch_df = df[df["Category"] == "Active Watch"].sort_values("Timing_Score", ascending=False)
        if watch_df.empty:
            st.info("No active watch setups now.")
        else:
            for _, row in watch_df.iterrows():
                card(row, key_prefix="watch")
    with tabs[2]:
        sell_df = df[df["Category"].isin(["Within Sell", "Near Sell"])].sort_values("Risk_Score", ascending=False)
        if sell_df.empty:
            st.info("No sell-risk setups now.")
        else:
            for _, row in sell_df.iterrows():
                card(row, key_prefix="sell")
    with tabs[3]:
        all_df = df.sort_values(["Category", "Timing_Score"], ascending=[True, False])
        for _, row in all_df.iterrows():
            card(row, key_prefix=f"all_{_}")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="ai_stock_mobile_results.csv", mime="text/csv", use_container_width=True)
