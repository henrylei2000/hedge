import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_vpt(df):
    """
    Calculate Volume Price Trend (VPT) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

    Returns:
    pd.DataFrame: DataFrame with an additional 'VPT' column.
    """
    df['Price_Change'] = df['Close'].pct_change()
    df['VPT'] = (df['Price_Change'] * df['Volume']).cumsum()
    return df

def generate_vpt_signals(df, vpt_window=20):
    """
    Generate buy and sell signals based on VPT and its moving average.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'VPT' column.
    vpt_window (int): The window size for the VPT moving average.

    Returns:
    pd.DataFrame: DataFrame with additional 'VPT_MA' and 'Signal' columns.
    """
    # Calculate VPT moving average
    df['VPT_MA'] = df['VPT'].rolling(window=vpt_window).mean()

    # Generate signals
    df['Signal'] = 0
    df['Signal'][df['VPT'] > df['VPT_MA']] = 1
    df['Signal'][df['VPT'] < df['VPT_MA']] = -1
    return df

# Fetch historical data for TQQQ over the past year
ticker = 'TQQQ'
data = yf.download(ticker, start='2024-05-01', end='2024-07-01')

# Calculate VPT
data = calculate_vpt(data)

# Generate VPT signals
data = generate_vpt_signals(data)

# Drop NaN values resulting from rolling mean calculation
data = data.dropna()

# Plotting the data
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

# Plot close price with signals
ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
buy_signals = data[data['Signal'] == 1].index
sell_signals = data[data['Signal'] == -1].index
ax1.plot(buy_signals, data['Close'][buy_signals], '^', markersize=10, color='green', label='Buy Signal')
ax1.plot(sell_signals, data['Close'][sell_signals], 'v', markersize=10, color='red', label='Sell Signal')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

# Plot VPT and its moving average
ax2.plot(data.index, data['VPT'], label='VPT', color='red')
ax2.plot(data.index, data['VPT_MA'], label='VPT Moving Average', color='green')
ax2.set_ylabel('VPT')
ax2.legend(loc='upper left')

plt.xlabel('Date')
plt.title(f'VPT and Buy/Sell Signals for {ticker}')
plt.show()

# Print the latest signals
print(f'Latest signals for {ticker}:')
print(data[['Close', 'Signal']].tail())
