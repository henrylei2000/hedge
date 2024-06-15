import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def analyze_seasonality(stock_symbol, start_date='2003-01-01', end_date='2023-01-01'):
    # Download historical stock price data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Check if data is fetched properly
    if stock_data.empty:
        print("No data fetched for the given stock symbol and date range.")
        return

    # Use the 'Close' price for the analysis
    stock_data['Date'] = stock_data.index
    stock_data = stock_data[['Date', 'Close']]

    # Set the 'Date' column as the index
    stock_data.set_index('Date', inplace=True)

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=365)

    # Plot the decomposition
    plt.figure(figsize=(14, 7))
    plt.subplot(411)
    plt.plot(stock_data['Close'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return decomposition


# Example usage
decomposition = analyze_seasonality('AAPL')
