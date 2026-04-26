import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Data Extraction", layout="wide")

st.title("Technical Analysis Indicator")
st.write("Extract stock market prices from Yahoo Finance using a ticker symbol.")

st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Enter Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if st.sidebar.button("Get Data"):

    # Create ticker object
    stock = yf.Ticker(ticker)

    # Download historical prices
    df = stock.history(start=start_date, end=end_date)

    # Validate data
    if df.empty:
        st.error("No data found. Check the ticker symbol or date range.")
        st.stop()

    st.success(f"Data successfully extracted for {ticker}")

    # -----------------------------
    # COMPANY INFORMATION (SAFE)
    # -----------------------------
    st.subheader("Company Information")

    try:
        info = stock.fast_info
    except Exception:
        st.error("Company information unavailable due to Yahoo Finance rate limits.")
        info = {}

    company_name = info.get("longName", "N/A")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market_cap = info.get("marketCap", "N/A")
    website = info.get("website", "N/A")

    st.write(f"**Company Name:** {company_name}")
    st.write(f"**Sector:** {sector}")
    st.write(f"**Industry:** {industry}")
    st.write(f"**Market Cap:** {market_cap}")
    st.write(f"**Website:** {website}")

    # -----------------------------
    # HISTORICAL DATA
    # -----------------------------
    st.subheader("Historical Data")
    st.dataframe(df)

    # -----------------------------
    # CLOSING PRICE CHART
    # -----------------------------
    st.subheader("Closing Price Chart")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    st.pyplot(fig)

    # -----------------------------
    # MOVING AVERAGES & TREND
    # -----------------------------
    st.subheader("Moving Averages and Trend Analysis")

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
    ax_ma.plot(df["Close"], label="Price")
    ax_ma.plot(df["MA20"], label="MA20")
    ax_ma.plot(df["MA50"], label="MA50")
    ax_ma.plot(df["MA200"], label="MA200")
    ax_ma.set_title("Moving Averages vs Price")
    ax_ma.set_xlabel("Date")
    ax_ma.set_ylabel("Price")
    ax_ma.legend()
    st.pyplot(fig_ma)

    trend = "N/A"

    if len(df) >= 200:
        current_price = df["Close"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]
        ma50 = df["MA50"].iloc[-1]
        ma200 = df["MA200"].iloc[-1]

        st.write(f"**Current Price:** {current_price:.2f}")
        st.write(f"**MA20:** {ma20:.2f}")
        st.write(f"**MA50:** {ma50:.2f}")
        st.write(f"**MA200:** {ma200:.2f}")

        if current_price > ma20 and current_price > ma50 and current_price > ma200:
            st.success("**Trend:** Upward trend")
            trend = "upward"
        elif current_price < ma20 and current_price < ma50 and current_price < ma200:
            st.error("**Trend:** Downward trend")
            trend = "downward"
        else:
            st.info("**Trend:** Mixed trend")
            trend = "mixed"
    else:
        st.warning("Not enough data for 200-day MA. Need at least 200 data points.")

    # -----------------------------
    # RSI
    # -----------------------------
    st.subheader("Relative Strength Index (RSI)")

    if len(df) >= 14:
        delta = df["Close"].diff(1)
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(com=13, adjust=False).mean()
        avg_loss = losses.ewm(com=13, adjust=False).mean()

        if avg_loss.iloc[-1] == 0:
            rs = 100
        else:
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]

        rsi = 100 - (100 / (1 + rs))
        st.write(f"**RSI (14-period):** {rsi:.2f}")

        if rsi < 30:
            st.info("**RSI Interpretation:** Oversold")
            rsi_state = "oversold"
        elif rsi > 70:
            st.error("**RSI Interpretation:** Overbought")
            rsi_state = "overbought"
        else:
            st.success("**RSI Interpretation:** Neutral")
            rsi_state = "neutral"
    else:
        st.warning("Not enough data to calculate RSI.")
        rsi_state = "N/A"

    # -----------------------------
    # TRADING SIGNAL
    # -----------------------------
    st.subheader("Trading Signal")

    if trend == "upward" and rsi_state == "oversold":
        signal = "Strong Buy"
        st.success(f"**Trading Signal:** {signal}")
    elif trend == "upward":
        signal = "Buy"
        st.success(f"**Trading Signal:** {signal}")
    elif trend == "downward" and rsi_state == "overbought":
        signal = "Strong Sell"
        st.error(f"**Trading Signal:** {signal}")
    elif trend == "downward":
        signal = "Sell"
        st.error(f"**Trading Signal:** {signal}")
    else:
        signal = "Hold"
        st.info(f"**Trading Signal:** {signal}")

    # -----------------------------
    # VOLATILITY
    # -----------------------------
    st.subheader("Volatility Analysis")

    VOL_PERIOD = 20

    if len(df) >= VOL_PERIOD:
        df["Daily_Return"] = df["Close"].pct_change()
        df["Rolling_Std"] = df["Daily_Return"].rolling(VOL_PERIOD).std()
        df["Annualized_Vol"] = df["Rolling_Std"] * np.sqrt(252)

        st.write("**Volatility Data (last 5 rows):**")
        st.dataframe(df[["Close", "Daily_Return", "Rolling_Std", "Annualized_Vol"]].tail())

        latest_vol = df["Annualized_Vol"].iloc[-1]

        def classify(vol):
            pct = vol * 100
            if pct > 40:
                return "High"
            elif pct >= 25:
                return "Medium"
            return "Low"

        st.write(f"**{VOL_PERIOD}-Day Volatility:** {latest_vol:.2%}")
        st.write(f"**Category:** {classify(latest_vol)}")
    else:
        st.warning(f"Not enough data for {VOL_PERIOD}-day volatility.")

    # -----------------------------
    # DOWNLOAD CSV
    # -----------------------------
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv"
    )
