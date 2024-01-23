import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def calculate_rsi(data, period=13):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_average_rsi(data, period=13, average_period=10):
    rsi = calculate_rsi(data, period)
    return rsi.rolling(window=average_period).mean()


def calculate_momentum_indicator(data, short_rsi_period=13, long_rsi_period=13, short_average_period=13, long_average_period=26):
    short_term_average_rsi = calculate_average_rsi(data, short_rsi_period, short_average_period)
    long_term_average_rsi = calculate_average_rsi(data, long_rsi_period, long_average_period)
    momentum_indicator = short_term_average_rsi - long_term_average_rsi
    return momentum_indicator


def calculate_sma(data, period=50):
    return data.rolling(window=period).mean()


def plot_price_and_momentum(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data available for {stock_symbol}.")
        return

    stock_data['Momentum'] = calculate_momentum_indicator(stock_data['Adj Close'])
    stock_data['SMA50'] = calculate_sma(stock_data['Adj Close'], period=50)
    momentum_sign = np.sign(stock_data['Momentum'])
    stock_data['Signal'] = momentum_sign.diff().fillna(0)
    stock_data['Positions'] = stock_data['Signal'].replace(-1, np.nan).ffill().fillna(0)

    # Calculate P/L
    stock_data['PnL'] = 0
    last_buy_price = None
    for i in range(len(stock_data)):
        if stock_data['Signal'][i] == 1:  # Buy
            last_buy_price = stock_data['Adj Close'][i]
        elif stock_data['Signal'][i] == -1 and last_buy_price is not None:  # Sell
            stock_data['PnL'][i] = stock_data['Adj Close'][i] - last_buy_price
            last_buy_price = None

    # Display trades summary
    trades_summary = stock_data[['Adj Close', 'Signal', 'Positions', 'PnL']][stock_data['Signal'] != 0]
    print("Trades Summary:")
    print(trades_summary)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot Price and SMA50
    ax1.plot(stock_data.index, stock_data['Adj Close'], label='Price', color='blue')
    ax1.plot(stock_data.index, stock_data['SMA50'], label='50-Day SMA', color='orange', linestyle='--')
    ax1.plot(stock_data.index[stock_data['Signal'] == 2], stock_data['Adj Close'][stock_data['Signal'] == 2], '^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stock_data.index[stock_data['Signal'] == -2], stock_data['Adj Close'][stock_data['Signal'] == -2], 'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot Custom Momentum Indicator
    ax2.plot(stock_data.index, stock_data['Momentum'], label='Momentum Indicator', color='green')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Momentum')
    ax2.legend()

    plt.title(f'{stock_symbol} Stock Price and Momentum Indicator')
    plt.show()


plot_price_and_momentum('QQQ', '2020-01-01', '2023-12-29')
