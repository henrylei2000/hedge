import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def calculate_price_volume_changes(df):
    """
    Calculate daily price change and volume change for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

    Returns:
    pd.DataFrame: DataFrame with additional 'Price_Change' and 'Volume_Change' columns.
    """
    # Calculate daily returns (price change)
    df['Price_Change'] = df['Close'].pct_change()
    # Calculate volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    return df


# Fetch historical data for AAPL over the past year
ticker = 'TQQQ'
data = yf.download(ticker, start='2024-05-01', end='2024-06-01')

# Calculate price and volume changes
data = calculate_price_volume_changes(data)

# Drop NaN values resulting from pct_change calculation
data = data.dropna()

# Plotting the data
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

# Plot price change
ax1.plot(data.index, data['Price_Change'], label='Price Change', color='blue')
ax1.set_ylabel('Price Change')
ax1.legend(loc='upper left')

# Plot volume change
ax2.plot(data.index, data['Volume_Change'], label='Volume Change', color='red')
ax2.set_ylabel('Volume Change')
ax2.legend(loc='upper left')

plt.xlabel('Date')
plt.title('Price Change and Volume Change Analysis for AAPL')
plt.show()

# Calculate correlation
correlation = data[['Price_Change', 'Volume_Change']].corr()
print('Correlation between Price Change and Volume Change for AAPL:')
print(correlation)
