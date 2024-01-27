import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Function to get historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, interval="15m", start=start_date, end=end_date)
    return stock_data

# Function to calculate MACD and generate buy/sell signals
def generate_signals(data):
    short_window = 10
    long_window = 21
    signal_window = 6

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

    return data

# Function to backtest the trading strategy
def backtest_strategy(data):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    shares_held = 0

    for index, row in data.iterrows():
        if row['Signal'] == 1 and balance > 0:
            # Buy signal
            shares_bought = balance // row['Close']
            position += row['Close'] * shares_bought
            balance -= row['Close'] * shares_bought
            shares_held += shares_bought
            if shares_bought:
                print(f"Bought at: ${row['Close']:.2f} x {shares_bought}")

        elif row['Signal'] == -1 and shares_held > 0:
            # Sell signal
            balance += row['Close'] * shares_held
            position -= row['Close'] * shares_held
            print(f"Sold at: ${row['Close']:.2f} x {shares_held}")
            print(f"Balance: ${balance:.2f}")
            shares_held = 0

    # Calculate final balance
    final_balance = balance + (shares_held * data['Close'].iloc[-1])

    return initial_balance, final_balance


# Define stock symbol and date range
ticker_symbol = 'QQQ'
start_date = '2024-01-22'
end_date = '2024-01-26'

# Fetch historical stock data
stock_data = get_stock_data(ticker_symbol, start_date, end_date)
stock_data.to_csv(f'./stock_data/{ticker_symbol}.csv')
# Generate signals
stock_data_with_signals = generate_signals(stock_data)

# Backtest the strategy
initial_balance, final_balance = backtest_strategy(stock_data_with_signals)

# Print results
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${final_balance:.2f}")

# Plotting the stock prices and signals
plt.figure(figsize=(12, 6))
plt.plot(stock_data_with_signals['Close'], label='Stock Price', linewidth=1)
plt.scatter(stock_data_with_signals.index[stock_data_with_signals['Signal'] == 1], stock_data_with_signals['Close'][stock_data_with_signals['Signal'] == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(stock_data_with_signals.index[stock_data_with_signals['Signal'] == -1], stock_data_with_signals['Close'][stock_data_with_signals['Signal'] == -1], marker='v', color='r', label='Sell Signal')
plt.title(f'{ticker_symbol} Stock Price with MACD Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
