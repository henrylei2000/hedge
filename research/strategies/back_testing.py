import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas as pd
import configparser
from macd_strategy import MACDStrategy


def download_from_yahoo(symbol, start_date="2023-10-27", end_date="2023-10-28"):
    # Download stock data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date)
    data.rename_axis('timestamp', inplace=True)
    data.rename(columns={'Close': 'close'}, inplace=True)
    return data


def download_from_alpaca(symbol):
    # Load Alpaca API credentials from configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Access configuration values
    api_key = config.get('settings', 'API_KEY')
    secret_key = config.get('settings', 'SECRET_KEY')

    # Initialize Alpaca API
    api = tradeapi.REST(api_key, secret_key, api_version='v2')

    # Convert start and end dates to RFC3339 format
    start_str = '2024-02-26'
    end_str = '2024-02-26'

    # Format start and end times for the filter
    start_time = pd.Timestamp(start_str + "T09:15:00", tz='America/New_York').isoformat()
    end_time = pd.Timestamp(end_str + "T16:00:00", tz='America/New_York').isoformat()  # Market closes at 4:00pm EST/EDT

    # Retrieve stock price data from Alpaca
    data = api.get_bars(symbol, '1Min', start=start_time, end=end_time).df

    if not data.empty:
        # Convert timestamp index to Eastern Timezone (EST)
        data.index = data.index.tz_convert('US/Eastern')

        # Filter rows between 9:30am and 4:00pm EST
        data = data.between_time('9:35', '15:55')

    return data


# Example usage
if __name__ == "__main__":
    # Download stock data
    stock = "TQQQ"
    stock_data = download_from_alpaca(stock)

    if not stock_data.empty:
        # Backtest MACD strategy
        macd_strategy = MACDStrategy(stock_data)
        macd_strategy.backtest()
        print(f"------- Total PnL Performance ------------ {macd_strategy.pnl:.2f}")
    else:
        print(f"------- Market is closed or the ticker is delisted.")
