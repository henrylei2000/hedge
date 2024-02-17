from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate MACD and generate buy/sell signals
def generate_signals(data):
    short_window = 3
    long_window = 6
    signal_window = 2

    # Calculate short-term and long-term exponential moving averages
    data['Short_MA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_MA'] = data['Close'].ewm(span=long_window, adjust=False).mean()

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

    data.dropna(subset=['Close'], inplace=True)
    return data


# Function to backtest the trading strategy
def backtest_strategy(data, seed):
    initial_balance = seed
    balance = initial_balance
    position = 0
    shares_held = 0
    trades = 0
    prev_signals = deque(maxlen=4)  # Keep track of the last 4 signals
    updated_signals = []  # Store updated signals

    for index, row in data.iterrows():
        signal = row['Signal']
        if row['Signal'] == 1 and balance > 0 and len(prev_signals) == 4 and prev_signals[-1] != -10 and prev_signals[-1] == 1:
            # Buy signal
            shares_bought = balance // row['Close']
            position += row['Close'] * shares_bought
            balance -= row['Close'] * shares_bought
            shares_held += shares_bought
            if shares_bought:
                signal = 10
                print(f"Bought at: ${row['Close']:.2f} x {shares_bought}")

        elif row['Signal'] == -1 and shares_held > 0 and prev_signals[-1] != 10 and prev_signals[-1] == -1:
            # Sell signal
            signal = -10
            trades += 1
            balance += row['Close'] * shares_held
            position -= row['Close'] * shares_held
            print(f"Sold at: ${row['Close']:.2f} x {shares_held}")
            print(f"Trade {trades} ------------- Balance: ${balance:.2f}")
            shares_held = 0

        updated_signals.append(signal)
        prev_signals.append(signal)

    # Calculate final balance
    final_balance = balance + (shares_held * data['Close'].iloc[-1])
    pnl_once = final_balance - initial_balance
    # Print results
    print(f"Initial Balance: ${initial_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
          f"\n----------------- PnL: ${pnl_once:.2f}")

    data['Signal'] = updated_signals

    return pnl_once


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
    formatter = MyFormatter(r.Datetime)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(np.arange(len(r)), r.Close, linewidth=1)
    ax.scatter(np.where(r.Signal == 1)[0], r.Close[r.Signal == 1], marker='^', color='g', label='Buy Signal')
    ax.scatter(np.where(r.Signal == -1)[0], r.Close[r.Signal == -1], marker='v', color='r', label='Sell Signal')
    ax.scatter(np.where(r.Signal == 10)[0], r.Close[r.Signal == 10], marker='o', color='g', label='Sell Signal')
    ax.scatter(np.where(r.Signal == -10)[0], r.Close[r.Signal == -10], marker='o', color='r', label='Sell Signal')
    fig.autofmt_xdate()
    plt.show()


def trade(interval, start, end, seed):

    profit = 0
    # Fetch historical stock data
    # stock_data = yf.download(ticker, interval=trade_interval, start=start_date, end=end_date)
    stock_data = yf.Ticker(ticker).history(interval=interval, start=start, end=end)
    if not stock_data.empty:
        stock_data.to_csv(f'./stock_data/{ticker}.csv')
        # Generate signals
        stock_data_with_signals = generate_signals(stock_data)
        # Backtest the strategy
        profit = backtest_strategy(stock_data_with_signals, seed)
        # draw_signals(stock_data_with_signals)

    return profit


# Define stock symbol and date range
ticker = 'SOXL'
pnl = 0
start = '2024-01-17'  # str, dt, int
end = '2024-02-18'
intraday = "5m"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
seed = 10000
mode = "daily"  # daily (without reset), batch, reset (with a reset balance daily)

if mode == "reset" or mode == "daily":
    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')

    # Initialize a variable to hold the current date
    current_date = start_date

    # Loop over dates
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        if mode == "reset":
            pnl += trade(intraday, current_date, next_date, seed)
        elif mode == "daily":
            pnl += trade(intraday, current_date, next_date, seed + pnl)
        current_date += timedelta(days=1)

    print(f"\n****** PnL ****** {pnl:.2f}")
else:
    trade(intraday, start, end, seed)


