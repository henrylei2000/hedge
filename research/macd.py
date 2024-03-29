from datetime import datetime, timedelta
from collections import deque
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import pandas as pd
import configparser

import numpy as np


def analyze_macd_derivative(macd_values):
    """
    Analyze the derivative of MACD values and determine significance.

    Parameters:
    - macd_values (list or numpy array): MACD values.

    Returns:
    - derivative (list): First-order finite difference of MACD values.
    - significance (str): Significance of the derivative (increasing, decreasing, or stable).
    """
    derivative = [macd_values[i] - macd_values[i-1] for i in range(1, len(macd_values))]

    if all(derivative_val > 0 for derivative_val in derivative):
        significance = "increasing"
    elif all(derivative_val < 0 for derivative_val in derivative):
        significance = "decreasing"
    else:
        significance = "stable"

    return derivative, significance

# Example MACD values
macd_values = [-8.372, -3.641, -4.047, -5.025, -2.672, -1.328, 0.215]

# Analyze the derivative
derivative, significance = analyze_macd_derivative(macd_values)
print("MACD Derivative:", derivative)
print("Significance:", significance)


def identify_outliers(data, current, percentile_threshold=0.15):
    """
    Identify outliers in the data array based on percentile thresholds.

    Parameters:
    - data (list or numpy array): Input data array.
    - percentile_threshold (float): Percentile threshold for identifying outliers.
                                    Default is 0.15 (15%).

    Returns:
    - outliers (list): List of outlier indices.
    """
    # Convert data to numpy array if it's not already
    data = np.array(data)

    # Calculate percentile thresholds
    positive_threshold = np.percentile(data, 100 - percentile_threshold * 100)
    negative_threshold = np.percentile(data, percentile_threshold * 100)

    if current > positive_threshold:
        return 'top'
    elif current < negative_threshold:
        return 'bottom'
    else:
        return 'meh'


# Function to calculate MACD and generate buy/sell signals
def generate_signals(data):
    short_window, long_window, signal_window = 3, 7, 2

    # Calculate short-term and long-term exponential moving averages
    data['Short_MA'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['Long_MA'] = data['close'].ewm(span=long_window, adjust=False).mean()

    # Calculate MACD line
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    # Calculate Signal line
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    # Generate Buy and Sell signals
    data['Signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

    # Buy signal: MACD crosses above Signal line
    data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1

    # Sell signal: MACD crosses below Signal line
    data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1

    data.dropna(subset=['close'], inplace=True)

    return data


# Function to backtest the trading strategy
def backtest_strategy(data, initial_balance):
    balance = initial_balance
    position = 0
    shares_held = 0
    trades = 0
    prev_signals = deque(maxlen=10)  # Keep track of the last 4 signals
    updated_signals = []  # Store updated signals

    for index, row in data.iterrows():
        signal = row['Signal']
        if shares_held > 0 and signal == -1:
            # Sell signal
            signal = -10
            trades += 1
            balance += row['close'] * shares_held
            position -= row['close'] * shares_held
            print(f"Sold at: ${row['close']:.2f} x {shares_held}")
            print(f"Trade {trades} ------------- Balance: ${balance:.2f}")
            print(f"----------------- {row['MACD']}")
            shares_held = 0

        elif balance > 0 and signal == 1:
            # Buy signal
            shares_bought = balance // row['close']
            position += row['close'] * shares_bought
            balance -= row['close'] * shares_bought
            shares_held += shares_bought
            if shares_bought:
                print(f"share bought: {shares_bought:.2f}")
                signal = 10
                print(f"Bought at: ${row['close']:.2f} x {shares_bought}")

        updated_signals.append(signal)
        prev_signals.append(signal)

    data['Signal'] = updated_signals

    # Calculate final balance
    final_balance = balance + (shares_held * data['close'].iloc[-1])
    # Print results
    print(f"Initial Balance: ${initial_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
          f"\n----------------- PnL: ${final_balance - initial_balance:.2f}")

    return final_balance - initial_balance


def draw_signals(signals):
    class MyFormatter:
        def __init__(self, dates, fmt='%Y-%m-%d'):
            self.dates = dates
            self.fmt = fmt

        def __call__(self, x, pos=0):
            'Return the label for time x at position pos'
            ind = int(np.round(x))
            if ind >= len(self.dates) or ind < 0:
                return ''

            return pd.to_datetime(self.dates[ind]).strftime(self.fmt)

    r = signals.to_records()
    formatter = MyFormatter(r.timestamp)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(np.arange(len(r)), r.close, linewidth=1)
    ax.scatter(np.where(r.Signal == 1)[0], r.close[r.Signal == 1], marker='^', color='g', label='Buy Signal')
    ax.scatter(np.where(r.Signal == -1)[0], r.close[r.Signal == -1], marker='v', color='r', label='Sell Signal')
    ax.scatter(np.where(r.Signal == 10)[0], r.close[r.Signal == 10], marker='o', color='g', label='Buy Signal')
    ax.scatter(np.where(r.Signal == -10)[0], r.close[r.Signal == -10], marker='o', color='r', label='Sell Signal')
    for i, (x, y) in enumerate(zip(np.where(r.Signal != 0)[0], r.close[r.Signal != 0])):
       ax.text(x, y, f"{100 * (r.MACD[i+1] - r.Signal_Line[i+1]):.2f}({r.MACD[i+1]*100:.3f})", fontsize=7, ha='right', va='bottom')
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def get_stock_data(ticker, interval, start, end, source='alpaca'):

    # Fetch historical stock data
    if source == 'yahoo':
        stock_data = yf.Ticker(ticker).history(interval=interval, start=start, end=end)
        stock_data.rename_axis('timestamp', inplace=True)
        stock_data.rename(columns={'Close': 'close'}, inplace=True)

    else:
        # Load Alpaca API credentials from configuration file
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Access configuration values
        api_key = config.get('settings', 'API_KEY')
        secret_key = config.get('settings', 'SECRET_KEY')

        # Initialize Alpaca API
        api = tradeapi.REST(api_key, secret_key, api_version='v2')

        # Convert start and end dates to RFC3339 format
        # start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        # end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        start_str = '2023-10-27T09:15:00-05:00'
        end_str = '2023-10-27T10:45:00-05:00'
        print(start_str)
        # Retrieve stock price data from Alpaca
        stock_data = api.get_bars(ticker, '1Min', start=start_str, end=end_str).df

        # Convert timestamp index to Eastern Timezone (EST)
        stock_data.index = stock_data.index.tz_convert('US/Eastern')

        # Filter rows between 9:30am and 4:00pm EST
        stock_data = stock_data.between_time('9:30', '15:55')

    return stock_data


def trade(stock_data, seed):
    profit = 0
    if not stock_data.empty:
        stock_data.to_csv(f'./stock_data/{ticker}.csv')
        # Generate signals
        stock_data_with_signals = generate_signals(stock_data)
        # Backtest the strategy
        profit = backtest_strategy(stock_data_with_signals, seed)
        draw_signals(stock_data_with_signals)

    return profit


# Define stock symbol and date range
ticker = 'TQQQ'
pnl = 0
start = '2024-02-21'  # str, dt, int
end = '2024-02-22'
intraday = "5m"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
seed = 10000
mode = "batch"  # daily (without reset), batch, reset (with a reset balance daily)

if mode == "reset" or mode == "daily":
    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')

    # Initialize a variable to hold the current date
    current_date = start_date

    # Loop over dates
    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        data = get_stock_data(ticker, intraday, current_date, next_date)
        if data.any and mode == "reset":
            pnl += trade(data, seed)
        elif data.any and mode == "daily":
            pnl += trade(data, seed + pnl)
        current_date += timedelta(days=1)

    print(f"\n****** PnL ****** {pnl:.2f}")
else:
    data = get_stock_data(ticker, intraday, start, end)
    trade(data, seed)
