import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
import configparser
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
            self.find_tops_bottoms()
            # self.train()
            self.plot()
        else:
            print("No data found, please verify symbol and date range.")

    def download(self, api="alpaca"):
        start_str = '2024-02-26'
        end_str = '2024-02-26'

        if api == 'yahoo':
            # Download stock data from Yahoo Finance
            data = yf.download(self.symbol, interval='1m', start=start_str, end=end_str)
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

            # Format start and end times for the filter
            start_time = pd.Timestamp(start_str + "T09:15:00", tz='America/New_York').isoformat()
            end_time = pd.Timestamp(end_str + "T16:01:00",
                                    tz='America/New_York').isoformat()  # Market closes at 4:00pm EST/EDT

            # Retrieve stock price data from Alpaca
            data = api.get_bars(self.symbol, '1Min', start=start_time, end=end_time).df

            if not data.empty:
                # Convert timestamp index to Eastern Timezone (EST)
                data.index = data.index.tz_convert('US/Eastern')

                # Filter rows between 9:30am and 4:00pm EST
                data = data.between_time('9:25', '16:00')
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
            position = 0
            if shares_held > 0 and row['position'] == -1:
                # Sell signal
                position = -1
                trades += 1
                balance += row['close'] * shares_held
                print(f"Sold at: ${row['close']:.2f} x {shares_held}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f} @ {row}")
                shares_held = 0

            elif balance > 0 and row['position'] == 1:
                # Buy signal
                shares_bought = balance // row['close']
                balance -= row['close'] * shares_bought
                shares_held += shares_bought
                if shares_bought:
                    print(f"share bought: {shares_bought:.2f}")
                    position = 1
                    print(f"Bought at: ${row['close']:.2f} x {shares_bought}")

            positions.append(position)

        self.data['position'] = positions

        # Calculate final balance
        final_balance = balance + (shares_held * self.data['close'].iloc[-1])
        # Print results
        self.pnl = final_balance - self.init_balance
        print(f"Initial Balance: ${self.init_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
              f"\n----------------- PnL: ${self.pnl:.2f}")

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

    def find_tops_bottoms(self):
        data = self.data

        # Find peaks (tops) and valleys (bottoms)
        peaks, _ = find_peaks(data['close'], prominence=0.1)  # Adjust prominence as needed
        valleys, _ = find_peaks(-data['close'], prominence=0.1)  # Adjust prominence as needed

        # Get the corresponding prices
        data['abnormality'] = 0

        # Set abnormality values for peaks and valleys
        data.loc[data.index[peaks], 'abnormality'] = 1
        data.loc[data.index[valleys], 'abnormality'] = -1

        data.to_csv(f'./{self.symbol}.csv')

    def train(self):
        import pandas as pd
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        data = self.data
        # Features (X) and target variable (y)
        X = data[['close', 'macd', 'signal_line']]
        y = data['abnormality']

        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=2)

        # Iterate through time series splits
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
            print(report)

            current = pd.DataFrame(self.data.iloc[-6]).transpose()
            print(current)
            prediction = model.predict(current[['close', 'macd', 'signal_line']])
            print("Predicted class label:", prediction)
