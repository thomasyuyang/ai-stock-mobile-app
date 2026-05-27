import html
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Stock Mobile", page_icon="📈", layout="centered")

DEFAULT_WATCHLIST = [
    "NVDA", "SMH", "VGT", "QQQM", "VOO",
    "AMD", "AVGO", "MSFT", "META", "GOOGL",
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


@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, period: str = "1y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - macd_signal
    df["Volume20Avg"] = df["Volume"].rolling(20).mean()
    return df.dropna()


def pct_distance(price, level):
    if level == 0 or pd.isna(level):
        return None
    return (price - level) / level * 100


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

    if (
        price > ema20
        and ema20 > ema50
        and 45 <= rsi <= 65
        and macd_hist > prev_macd_hist
        and -1 <= distance_to_ema20 <= 3
    ):
        category = "Within Buy"
    elif (
        ema20 > ema50
        and 40 <= rsi <= 68
        and -3 <= distance_to_ema20 <= 5
    ):
        category = "Near Buy"

    if price < ema20 and macd_hist < 0 and rsi < 45:
        category = "Within Sell"
    elif (
        price > ema20
        and distance_to_ema20 is not None
        and distance_to_ema20 >= 8
        and (rsi >= 70 or macd_hist < prev_macd_hist)
    ):
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
    result.update(links_for(ticker))
    return result


def card(row):
    ticker = html.escape(str(row["Ticker"]))
    category = html.escape(str(row["Category"]))
    visual_status_text = html.escape(str(row["Visual_Status"]))
    notes = html.escape(str(row["Notes"]))
    background = card_background(row["Category"], row["Score"])

    st.markdown(
        f'''
        <div style="border:1px solid #d9d9d9;border-radius:18px;padding:14px;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);background-color:{background};">
            <h3 style="margin:0 0 8px 0;">{row['Emoji']} {ticker} — {visual_status_text}</h3>
            <p style="margin:6px 0;"><b>Setup:</b> {category}</p>
            <p style="margin:6px 0;"><b>Price:</b> ${row['Price']} &nbsp; <b>Score:</b> {row['Score']}/9 &nbsp; <b>Grade:</b> {row['Grade']}</p>
            <p style="margin:6px 0;"><b>RSI:</b> {row['RSI']} &nbsp; <b>EMA20:</b> {row['EMA20']} &nbsp; <b>EMA50:</b> {row['EMA50']}</p>
            <p style="margin:6px 0;"><b>Dist to EMA20:</b> {row['Distance_to_EMA20_%']}% &nbsp; <b>MACD Hist:</b> {row['MACD_Hist']} &nbsp; <b>Vol Ratio:</b> {row['Volume_Ratio']}</p>
            <p style="margin:6px 0;"><b>Notes:</b> {notes}</p>
            <p style="margin:8px 0 0 0;">
                <a href="{row['Yahoo Chart']}" target="_blank">Yahoo Chart</a>
                &nbsp;|&nbsp;
                <a href="{row['Yahoo Quote']}" target="_blank">Yahoo Quote</a>
                &nbsp;|&nbsp;
                <a href="{row['Fidelity']}" target="_blank">Fidelity</a>
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


st.title("📈 AI Stock Mobile")
st.caption("Mobile-friendly AI stock/ETF setup scanner and analyzer")
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
    watchlist_text = st.text_area(
        "Tickers separated by commas",
        value=", ".join(DEFAULT_WATCHLIST),
        height=120,
    )

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
    st.session_state["mobile_last_scan"] = (
        datetime.now(ZoneInfo("America/Chicago"))
        .strftime("%Y-%m-%d %I:%M:%S %p %Z")
    )

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
        buy_df = df[df["Category"].isin(["Within Buy", "Near Buy"])].sort_values("Score", ascending=False)
        if buy_df.empty:
            st.info("No buy setups now.")
        else:
            for _, row in buy_df.iterrows():
                card(row)

    with tabs[1]:
        watch_df = df[df["Category"] == "Active Watch"].sort_values("Score", ascending=False)
        if watch_df.empty:
            st.info("No active watch setups now.")
        else:
            for _, row in watch_df.iterrows():
                card(row)

    with tabs[2]:
        sell_df = df[df["Category"].isin(["Within Sell", "Near Sell"])].sort_values("Score", ascending=True)
        if sell_df.empty:
            st.info("No sell-risk setups now.")
        else:
            for _, row in sell_df.iterrows():
                card(row)

    with tabs[3]:
        all_df = df.sort_values(["Category", "Score"], ascending=[True, False])
        for _, row in all_df.iterrows():
            card(row)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="ai_stock_mobile_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
