import streamlit as st
import pandas as pd
import yfinance as yf

st.write("""
# Simple Stock Market
Google
""")

tickerSymbol="GOOGL"
tickerData=yf.Ticker(tickerSymbol)
tickerDf=tickerData.history(period="5D")

st.line_chart(tickerDf.Close)