import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
import configparser
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd

class Strategy:
    def __init__(self, symbol='TQQQ', open='2024-03-01 09:30', close='2024-03-01 16:00'):  # QQQ, SPY, DIA
        self.symbol = symbol
        self.reference = False
        self.start = pd.Timestamp(open, tz='America/New_York').tz_convert('UTC')
        self.end = pd.Timestamp(close, tz='America/New_York').tz_convert('UTC')
        self.data = None
        self.qqq = None
        self.spy = None
        self.dia = None
        self.pnl = 0.00
        self.trades = 0
        self.init_balance = 10000
        self.num_buckets = 1

    def backtest(self):
        # prediction = self.predict()
        if True:
            if self.download():
                self.sanitize()
                self.signal()
                self.bucket_trade()
                self.plot()
                return
            else:
                print("No data found, please verify symbol and date range.")

    def predict(self):
        """
        Predicts the next day's peak and valley based on one year of daily prices and linear regression.

        Parameters:
        - ticker: The stock ticker symbol (default is 'TQQQ').

        Returns:
        - A dictionary containing the predicted peak and valley prices for the next day.
        """
        day = self.start.strftime('%Y-%m-%d')
        end_date = pd.to_datetime(day) - pd.DateOffset(days=1)
        print(end_date)
        start_date = end_date - pd.DateOffset(months=3)

        # Download one year of daily price data
        data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d')
        prices = data['Close']

        # Identify peaks and valleys
        peaks, _ = find_peaks(prices, distance=5, prominence=0.5)
        valleys, _ = find_peaks(-prices, distance=5, prominence=0.5)

        # Perform linear regression on peaks
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)

        # Perform linear regression on valleys
        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]
        a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)

        # Predict the next day's peak and valley
        next_day_index = len(prices)
        predicted_peak = a_peaks * next_day_index + b_peaks
        predicted_valley = a_valleys * next_day_index + b_valleys

        # Indicate whether the next day is predicted to be a peak or a valley
        if predicted_peak > predicted_valley:
            next_day_prediction = 'Peak'
        else:
            next_day_prediction = 'Valley'

            # Determine the type of the last recognized abnormal day
        last_peak_index = peak_indices[-1] if len(peak_indices) > 0 else -1
        last_valley_index = valley_indices[-1] if len(valley_indices) > 0 else -1

        if last_peak_index > last_valley_index:
            last_abnormal_day = 'Peak'
        else:
            last_abnormal_day = 'Valley'

        # Indicate whether the next day is predicted to be a peak or a valley based on the last abnormal day
        if last_abnormal_day == 'Peak':
            next_day_prediction = 'Valley'
        else:
            next_day_prediction = 'Peak'

        prediction = {
            'Next Day Index': next_day_index,
            'Predicted Peak': predicted_peak,
            'Predicted Valley': predicted_valley,
            'Prediction Type': next_day_prediction
        }
        print(prediction)
        return prediction

    def download(self, api="alpaca"):

        if api == 'yahoo':
            # Download stock data from Yahoo Finance
            self.data = yf.download(self.symbol, interval='1m', start=self.start, end=self.end)
            dataset = [self.data]
            if self.reference:
                self.qqq = yf.download('QQQ', interval='1m', start=self.start, end=self.end)
                self.spy = yf.download('SPY', interval='1m', start=self.start, end=self.end)
                self.dia = yf.download('DIA', interval='1m', start=self.start, end=self.end)
                dataset += [self.qqq, self.spy, self.dia]
            for data in [self.data]:
                if not data.empty:
                    data.rename_axis('timestamp', inplace=True)
                    data.rename(columns={'Close': 'close'}, inplace=True)
                    data.rename(columns={'High': 'high'}, inplace=True)
                    data.rename(columns={'Low': 'low'}, inplace=True)
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
            data = api.get_bars(self.symbol, '1Min', start=self.start.isoformat(), end=self.end.isoformat()).df

            if not data.empty:
                # Convert timestamp index to Eastern Timezone (EST)
                data.index = data.index.tz_convert('US/Eastern')
                # Filter rows between 9:30am and 4:00pm EST
                data = data.between_time('9:30', '15:59')
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
              f"\n-------{self.symbol}---------- PnL: ${self.pnl:.2f}")

    def bucket_trade(self):
        initial_balance = self.init_balance
        num_buckets = self.num_buckets  # 4
        initial_bucket_value = initial_balance / num_buckets
        buckets = [{'in_use': False, 'shares': 0, 'buy_price': 0, 'bucket_value': initial_bucket_value} for _ in
                   range(num_buckets)]
        total_balance = initial_balance  # Updated to track total balance across all buckets
        if 'position' not in self.data.columns:
            self.data['position'] = self.data['signal']
        stock_data = self.data
        for i, row in stock_data.iterrows():
            price = row['close']  # Assuming the 'close' column has the stock price
            position = row['position']

            if position == 1:  # Buy signal
                for bucket in buckets:
                    if not bucket['in_use']:
                        bucket['in_use'] = True
                        bucket['shares'] = bucket['bucket_value'] / price
                        bucket['buy_price'] = price
                        # print(f"BUY  ${price:.3f} x {bucket['shares']:.2f}  @{i}")
                        self.trades += 1
                        break  # Exit after finding the first available bucket

            elif position == -1:  # Sell signal
                for bucket in buckets:
                    if bucket['in_use']:
                        # Calculate the value after selling shares
                        # print(f"SELL ${price:.3f} x {bucket['shares']:.2f}  @{i}")
                        sell_value = bucket['shares'] * price
                        # Calculate PnL for this bucket
                        pnl = sell_value - bucket['bucket_value']
                        total_balance += pnl  # Update total balance
                        # Reset bucket for the next trade
                        bucket['in_use'] = False
                        bucket['shares'] = 0
                        bucket['buy_price'] = 0
                        bucket['bucket_value'] = sell_value  # Update bucket value with the result of the trade
                        break  # Assume one sell signal sells the shares from one bucket only

        # Calculate the final balance by adding up the remaining bucket values
        final_balance = sum(bucket['bucket_value'] for bucket in buckets if not bucket['in_use']) + \
                        sum(bucket['shares'] * price for bucket in buckets if bucket['in_use'])
        total_pnl = final_balance - initial_balance
        self.pnl = total_pnl
        # print(f"---{self.symbol}----- Total PnL Performance ------------ {self.pnl:.2f}")
        return total_pnl

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

        fig, ax = plt.subplots(figsize=(18, 6))
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(np.arange(len(r)), r.close, linewidth=1)
        ax.scatter(np.where(r.signal == 1)[0], r.close[r.signal == 1], marker='^', color='g', label='Buy Signal')
        ax.scatter(np.where(r.signal == -1)[0], r.close[r.signal == -1], marker='v', color='r', label='Sell Signal')
        ax.scatter(np.where(r.position == 1)[0], r.close[r.position == 1], marker='o', color='g', alpha=.5, s=120,
                   label='Buy')
        ax.scatter(np.where(r.position == -1)[0], r.close[r.position == -1], marker='o', color='r', alpha=.5, s=120,
                   label='Sell')
        # for i, (x, y) in enumerate(zip(np.where(r.signal != 0)[0], r.close[r.signal != 0])):
        #     ax.text(x, y, f"{r.close[i]:.2f}", fontsize=7, ha='right', va='bottom')
        fig.autofmt_xdate()
        # fig.tight_layout()
        plt.title(f"{self.symbol}")
        plt.show()

