import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # Import the regular expression library for input validation
from datetime import datetime

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Function to provide buy and sell signals
def get_signals(rsi_values, buy_threshold=30, sell_threshold=70):
    signals = []

    for rsi in rsi_values:
        if rsi < buy_threshold:
            signals.append("Buy")
        elif rsi > sell_threshold:
            signals.append("Sell")
        else:
            signals.append("Hold")

    return signals

# Function to validate and parse the time frame input
def validate_time_frame(time_frame):
    # Use regular expressions to validate the format (e.g., 1h, 1d, 1wk, 1mo)
    if re.match(r'^\d+[hdwmo]$', time_frame):
        return True
    return False

# Function to validate and parse the date input
def validate_date(date_string):
    try:
        date = pd.Timestamp(date_string)
        return True, date
    except ValueError:
        return False, None

# Function to plot RSI within the specified time frame and date range
def plot_rsi(stock_symbol, time_frame, start_date, end_date, draw_graph=True):
    if not validate_time_frame(time_frame):
        print("Invalid time frame format. Please use a valid format like 1h, 1d, 1wk, 1mo.")
        return

    if not (validate_date(start_date) and validate_date(end_date)):
        print("Invalid date format. Please use a valid date format (e.g., YYYY-MM-DD).")
        return

    try:
        # Determine the interval based on the time frame
        if 'h' in time_frame:
            interval = '1h' if time_frame == '1h' else '2h'
        elif 'd' in time_frame:
            interval = '1d'
        elif 'w' in time_frame:
            interval = '1wk'
        elif 'm' in time_frame:
            interval = '1mo'
        else:
            print("Invalid time frame format. Please use a valid format like 1h, 1d, 1wk, 1mo.")
            return

        # Download historical stock data with the determined interval and date range
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval=interval)

        if stock_data.empty:
            print(f"No data available for {stock_symbol} with a {time_frame} time frame.")
            return

        # Calculate RSI with a period of 14 (you can adjust this as needed)
        rsi_period = 14
        stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'], period=rsi_period)

        # Get buy and sell signals
        buy_threshold = 30
        sell_threshold = 70
        stock_data['Signal'] = get_signals(stock_data['RSI'], buy_threshold, sell_threshold)

        # Create a figure and axis for the price chart
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the stock's adjusted closing price
        ax1.plot(stock_data.index, stock_data['Adj Close'], label='Price', color='blue')

        # Create a second axis for RSI and overlay it on the price chart
        ax2 = ax1.twinx()
        ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='red', linestyle='--')

        # Set labels and legends for both axes
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax2.set_ylabel('RSI', color='red')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Add horizontal lines for buy and sell thresholds
        ax2.axhline(y=buy_threshold, color='green', linestyle='--', label=f'Buy Threshold ({buy_threshold})')
        ax2.axhline(y=sell_threshold, color='red', linestyle='--', label=f'Sell Threshold ({sell_threshold})')

        if draw_graph:
            # Show the plot
            plt.title(f'{stock_symbol} Stock Price and RSI ({time_frame} period)')
            plt.show()

        # Print the DataFrame with RSI and signals
        print(stock_data[['Adj Close', 'RSI', 'Signal']])

        # Calculate and print the buy/sell/hold signal for the daily RSI
        daily_rsi = calculate_rsi(stock_data['Adj Close'], period=14)
        daily_signal = get_signals(daily_rsi, buy_threshold, sell_threshold)
        print(f"Daily RSI Signal: {daily_signal[-1]}")  # Print the signal for the last data point

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Input the stock symbol
stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")

# Input the time frame for RSI (e.g., 1h, 1d, 1wk, 1mo)
time_frame = input("Enter the time frame for RSI (e.g., 1h, 1d, 1wk, 1mo): ")

# Calculate and print the buy/sell/hold signal for the daily RSI
# You can adjust the start and end dates as needed
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = datetime.today().strftime('%Y-%m-%d')
plot_rsi(stock_symbol, "1d", start_date, end_date, draw_graph=False)

# Ask the user whether to draw the RSI graph
draw_graph_option = input("Do you want to draw the RSI graph? (y/n): ")
if draw_graph_option.lower() == 'y':
    plot_rsi(stock_symbol, time_frame, start_date, end_date)
