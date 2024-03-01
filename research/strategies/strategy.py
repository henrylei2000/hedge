import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
import configparser
import matplotlib.pyplot as plt


class Strategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None
        self.pnl = 0.00
        self.init_balance = 10000

    def backtest(self):
        if self.download():
            self.sanitize()
            self.signal()
            self.trade()
            self.plot()
            return
        else:
            print("No data found, please verify symbol and date range.")

    def download(self, api="alpaca"):
        start_str = '2024-02-29 9:30'
        end_str = '2024-02-29 15:00'
        start_time = pd.Timestamp(start_str, tz='America/New_York').tz_convert('UTC')
        end_time = pd.Timestamp(end_str, tz='America/New_York').tz_convert('UTC')

        if api == 'yahoo':
            # Download stock data from Yahoo Finance
            data = yf.download(self.symbol, interval='1m', start=start_time, end=end_time)
            if not data.empty:
                data.rename_axis('timestamp', inplace=True)
                data.rename(columns={'Close': 'close'}, inplace=True)
                self.data = data
            else:
                return False

        elif api == 'alpaca':
            # Load Alpaca API credentials from configuration file
            config = configparser.ConfigParser()
            config.read('config.ini')
            # Access configuration values
            api_key = config.get('settings', 'API_KEY')
            secret_key = config.get('settings', 'SECRET_KEY')
            # Initialize Alpaca API
            api = tradeapi.REST(api_key, secret_key, api_version='v2')

            # Retrieve stock price data from Alpaca
            data = api.get_bars(self.symbol, '1Min', start=start_time.isoformat(), end=end_time.isoformat()).df

            if not data.empty:
                # Convert timestamp index to Eastern Timezone (EST)
                data.index = data.index.tz_convert('US/Eastern')
                # Filter rows between 9:30am and 4:00pm EST
                data = data.between_time('9:30', '16:00')
                if not data.empty:
                    self.data = data
                else:
                    return False
            else:
                return False

        return True

    def sanitize(self):
        pass

    def signal(self):
        pass

    def trade(self):
        balance = self.init_balance
        shares_held = 0
        trades = 0
        positions = []  # Store updated signals
        if 'position' not in self.data.columns:
            self.data['position'] = self.data['signal']
        for index, row in self.data.iterrows():
            # position = 0
            if shares_held > 0 and row['position'] == -1:
                # Sell signal
                position = -1
                trades += 1
                balance += row['close'] * shares_held
                print(f"Sold at: ${row['close']:.2f} x {shares_held}  @{index}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f} [macd {row['macd']*100:.3f}]")
                shares_held = 0

            elif balance > 0 and row['position'] == 1:
                # Buy signal
                shares_bought = balance // row['close']
                balance -= row['close'] * shares_bought
                shares_held += shares_bought
                if shares_bought:
                    print(f"share bought: {shares_bought:.2f}")
                    position = 1
                    print(f"Bought at: ${row['close']:.2f} x {shares_bought}  @{index} [macd {row['macd']*100:.3f}]")

            # positions.append(position)

        # self.data['position'] = positions

        # Calculate final balance
        final_balance = balance + (shares_held * self.data['close'].iloc[-1])
        # Print results
        self.pnl = final_balance - self.init_balance
        print(f"Initial Balance: ${self.init_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
              f"\n----------------- PnL: ${self.pnl:.2f}")

    def bucket_trade(stock_data, initial_balance=10000):
        # Initial setup
        num_buckets = 3
        bucket_value = initial_balance / num_buckets
        buckets_in_use = 0  # Tracks how many buckets are currently invested
        balance = initial_balance
        shares_held = 0  # Total shares held

        for i, row in stock_data.iterrows():
            price = row['close']  # Assuming the 'close' column has the stock price
            position = row['position']

            # Buy signal and we have buckets available
            if position == 1 and buckets_in_use < num_buckets:
                # Calculate how many shares to buy with one bucket
                shares_to_buy = bucket_value / price
                shares_held += shares_to_buy
                buckets_in_use += 1
                balance -= bucket_value

            # Sell signal and we have shares to sell
            elif position == -1 and shares_held > 0 and buckets_in_use > 0:
                # Assume we sell shares bought with one bucket (evenly distributing shares across buckets)
                shares_to_sell = shares_held / buckets_in_use
                balance += shares_to_sell * price
                shares_held -= shares_to_sell
                buckets_in_use -= 1

        # Final balance after selling any remaining shares at the last price
        if shares_held > 0:
            balance += shares_held * stock_data.iloc[-1]['close']
            shares_held = 0

        return balance, shares_held

    def plot(self):
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

        r = self.data.to_records()
        formatter = MyFormatter(r.timestamp)

        fig, ax = plt.subplots(figsize=(24, 6))
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(np.arange(len(r)), r.close, linewidth=1)
        ax.scatter(np.where(r.signal == 1)[0], r.close[r.signal == 1], marker='^', color='g', label='Buy Signal')
        ax.scatter(np.where(r.signal == -1)[0], r.close[r.signal == -1], marker='v', color='r', label='Sell Signal')
        ax.scatter(np.where(r.position == 1)[0], r.close[r.position == 1], marker='o', color='g', alpha=.5, s=120,
                   label='Buy')
        ax.scatter(np.where(r.position == -1)[0], r.close[r.position == -1], marker='o', color='r', alpha=.5, s=120,
                   label='Sell')
        for i, (x, y) in enumerate(zip(np.where(r.signal != 0)[0], r.close[r.signal != 0])):
            if i < len(r.close) - 1:
                ax.text(x, y, f"{r.close[i + 1] * 1:.2f}", fontsize=7, ha='right', va='bottom')
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

