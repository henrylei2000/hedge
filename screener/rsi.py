import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate RSI
def calculate_rsi(data, period=13):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Function to calculate Simple Moving Average
def calculate_sma(data, period=21):
    return data.rolling(window=period).mean()


# Function to plot RSI and Price
def plot_rsi_and_price(stock_symbol, start_date, end_date, rsi_period=13, buy_threshold=35, sell_threshold=65, sma_period=21):
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data available for {stock_symbol}.")
            return

        stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'], period=rsi_period)
        stock_data['SMA'] = calculate_sma(stock_data['Adj Close'], period=sma_period)

        # Initialize trade signals and positions
        stock_data['Signal'] = 0  # No position
        stock_data['Positions'] = 0
        in_position = False
        buy_price = 0

        for i in range(1, len(stock_data)):
            if not in_position and stock_data['RSI'][i] < buy_threshold:
                stock_data['Signal'][i] = 1  # Buy signal
                in_position = True
                buy_price = stock_data['Adj Close'][i]
            elif in_position and (stock_data['RSI'][i] > sell_threshold): # or stock_data['Adj Close'][i] < buy_price
                stock_data['Signal'][i] = -1  # Sell signal
                in_position = False

        # Calculate positions
        stock_data['Positions'] = stock_data['Signal'].cumsum()

        # Calculate Profit/Loss
        stock_data['PnL'] = (stock_data['Adj Close'].diff() * stock_data['Positions'].shift()).cumsum()

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot Price, SMA, and Buy/Sell signals
        ax1.plot(stock_data.index, stock_data['Adj Close'], label='Price', color='blue')
        ax1.plot(stock_data.index, stock_data['SMA'], label='21-Day SMA', color='orange', linestyle='--')
        ax1.plot(stock_data.index[stock_data['Signal'] == 1], stock_data['Adj Close'][stock_data['Signal'] == 1], '^',
                 markersize=10, color='g', label='Buy Signal')
        ax1.plot(stock_data.index[stock_data['Signal'] == -1], stock_data['Adj Close'][stock_data['Signal'] == -1], 'v',
                 markersize=10, color='r', label='Sell Signal')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot RSI
        ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='red', linestyle='--')
        ax2.axhline(y=buy_threshold, color='green', linestyle='--')
        ax2.axhline(y=sell_threshold, color='red', linestyle='--')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()

        plt.title(f'{stock_symbol} Stock Price, RSI, and Trading Signals')
        plt.show()

        # Print summary of trades and PnL
        print("Trades Summary:")
        print(stock_data[['Adj Close', 'Signal', 'Positions', 'PnL']].loc[stock_data['Signal'] != 0])

    except Exception as e:
        print(f"An error occurred: {str(e)}")


plot_rsi_and_price('QQQ', '2023-01-01', '2023-12-29')
