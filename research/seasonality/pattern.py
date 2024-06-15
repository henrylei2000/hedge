import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def analyze_seasonality(stock_symbol, start_date='1993-01-01', end_date='2023-01-01'):
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

    # Calculate daily returns
    stock_data['Return'] = stock_data['Close'].pct_change()

    # Resample to monthly and weekly data
    monthly_data = stock_data['Return'].resample('M').mean()
    weekly_data = stock_data['Return'].resample('W').mean()

    # Aggregate data to find average monthly and daily patterns
    stock_data['Month'] = stock_data.index.month
    stock_data['Weekday'] = stock_data.index.weekday

    monthly_patterns = stock_data.groupby('Month')['Return'].mean()
    weekly_patterns = stock_data.groupby('Weekday')['Return'].mean()

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(stock_data['Close'].dropna(), model='multiplicative', period=365)

    # Plot the decomposition
    plt.figure(figsize=(14, 10))
    plt.subplot(511)
    plt.plot(stock_data['Close'], label='Original')
    plt.legend(loc='best')
    plt.subplot(512)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(513)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(514)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='best')
    plt.subplot(515)
    plt.plot(stock_data['Return'], label='Daily Returns')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot monthly patterns
    plt.figure(figsize=(14, 5))
    monthly_patterns.plot(kind='bar')
    plt.title('Average Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(12),
               labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.grid(True)
    plt.show()

    # Plot weekly patterns
    plt.figure(figsize=(14, 5))
    weekly_patterns.plot(kind='bar')
    plt.title('Average Weekly Returns')
    plt.xlabel('Weekday')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
    plt.grid(True)
    plt.show()

    return {
        'decomposition': decomposition,
        'monthly_patterns': monthly_patterns,
        'weekly_patterns': weekly_patterns
    }


# Example usage
result = analyze_seasonality('AAPL')
print(result)