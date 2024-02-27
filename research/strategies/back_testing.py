import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas as pd
import configparser
from macd_strategy import MACDStrategy


def download_from_yahoo(symbol, start_date, end_date):
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
    # start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    # end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    start_str = '2023-10-27T09:15:00-05:00'
    end_str = '2023-10-27T10:45:00-05:00'
    print(start_str)
    # Retrieve stock price data from Alpaca
    data = api.get_bars(symbol, '1Min', start=start_str, end=end_str).df

    # Convert timestamp index to Eastern Timezone (EST)
    data.index = data.index.tz_convert('US/Eastern')

    # Filter rows between 9:30am and 4:00pm EST
    data = data.between_time('9:30', '15:55')

    return data


# Example usage
if __name__ == "__main__":
    # Download stock data
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    stock_data = download_from_alpaca(symbol)

    # Backtest MACD strategy
    macd_strategy = MACDStrategy(stock_data)
    macd_strategy.backtest()
    print(macd_strategy.pnl)
