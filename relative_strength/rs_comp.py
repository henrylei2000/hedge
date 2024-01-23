import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

def plot_stock_comparison(stock_a, stock_b, start_date, end_date):
    """
    Plot the price movements of two stocks and their relative strength.
    """
    # Fetch stock data
    data_a = fetch_stock_data(stock_a, start_date, end_date)
    data_b = fetch_stock_data(stock_b, start_date, end_date)

    # Calculate relative strength
    relative_strength = data_a / data_b

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot stock A and B prices
    axes[0].plot(data_a, label=f'{stock_a} Price')
    axes[0].plot(data_b, label=f'{stock_b} Price')
    axes[0].set_title('Stock Prices')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].legend()

    # Plot relative strength with dynamic label
    relative_strength_label = f'Relative Strength ({stock_a}/{stock_b})'
    axes[1].plot(relative_strength, label=relative_strength_label)
    axes[1].set_title('Relative Strength Line')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Relative Strength')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Example usage
plot_stock_comparison('spy', 'qqq', '2023-12-01', '2023-12-29')
