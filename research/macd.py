from datetime import datetime, timedelta
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
def backtest_strategy(data):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    shares_held = 0
    trades = 0
    first_sell_signal_occurred = False  # Flag to track the first sell signal

    for index, row in data.iterrows():
        if row['Signal'] == 1 and balance > 0:
            # Buy signal
            shares_bought = balance // row['Close']
            position += row['Close'] * shares_bought
            balance -= row['Close'] * shares_bought
            shares_held += shares_bought
            if shares_bought:
                print(f"Bought at: ${row['Close']:.2f} x {shares_bought}")
            first_sell_signal_occurred = False
        elif row['Signal'] == -1 and shares_held > 0:
            # Sell signal
            if first_sell_signal_occurred:
                trades += 1
                balance += row['Close'] * shares_held
                position -= row['Close'] * shares_held
                print(f"Sold at: ${row['Close']:.2f} x {shares_held}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f}")
                shares_held = 0
            else:
                first_sell_signal_occurred = True

    # Calculate final balance
    final_balance = balance + (shares_held * data['Close'].iloc[-1])

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
    formatter = MyFormatter(r.Datetime)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(np.arange(len(r)), r.Close, linewidth=1)
    ax.scatter(np.where(r.Signal == 1)[0], r.Close[r.Signal == 1], marker='^', color='g', label='Buy Signal')
    ax.scatter(np.where(r.Signal == -1)[0], r.Close[r.Signal == -1], marker='v', color='r', label='Sell Signal')
    fig.autofmt_xdate()
    plt.show()


# Define stock symbol and date range
ticker = 'TQQQ'
pnl = 0
start_date = datetime.strptime('2024-01-16', '%Y-%m-%d')  # str, dt, int
end_date = datetime.strptime('2024-02-15', '%Y-%m-%d')
intraday = "5m"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

# Fetch historical stock data
# stock_data = yf.download(ticker, interval=trade_interval, start=start_date, end=end_date)

stock_data = yf.Ticker(ticker).history(interval=intraday, start=start_date, end=end_date)
if not stock_data.empty:
    stock_data.to_csv(f'./stock_data/{ticker}.csv')
    # Generate signals
    stock_data_with_signals = generate_signals(stock_data)
    # Backtest the strategy
    backtest_strategy(stock_data_with_signals)
    draw_signals(stock_data_with_signals)
