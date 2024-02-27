from datetime import datetime, timedelta
from collections import deque
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import pandas as pd
import configparser

import numpy as np


class Strategy:
    def __init__(self, data):
        self.data = data
        self.pnl = 0.00
        self.init_balance = 10000

    def backtest(self):
        self.sanitize()
        self.signal()
        self.trade()
        self.plot()

    def sanitize(self):
        pass

    def signal(self):
        pass

    def trade(self):
        balance = self.init_balance
        position = 0
        shares_held = 0
        trades = 0
        prev_signals = deque(maxlen=10)  # Keep track of the last 4 signals
        updated_signals = []  # Store updated signals

        for index, row in self.data.iterrows():
            signal = row['Signal']
            if shares_held > 0 and signal == -1:
                # Sell signal
                signal = -10
                trades += 1
                balance += row['close'] * shares_held
                position -= row['close'] * shares_held
                print(f"Sold at: ${row['close']:.2f} x {shares_held}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f}")
                print(f"----------------- {row['MACD']}")
                shares_held = 0

            elif balance > 0 and signal == 1:
                # Buy signal
                shares_bought = balance // row['close']
                position += row['close'] * shares_bought
                balance -= row['close'] * shares_bought
                shares_held += shares_bought
                if shares_bought:
                    print(f"share bought: {shares_bought:.2f}")
                    signal = 10
                    print(f"Bought at: ${row['close']:.2f} x {shares_bought}")

            updated_signals.append(signal)
            prev_signals.append(signal)

        self.data['Signal'] = updated_signals

        # Calculate final balance
        final_balance = balance + (shares_held * self.data['close'].iloc[-1])
        # Print results
        print(f"Initial Balance: ${self.init_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
              f"\n----------------- PnL: ${final_balance - self.init_balance:.2f}")

        self.pnl = final_balance - self.init_balance

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

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(np.arange(len(r)), r.close, linewidth=1)
        ax.scatter(np.where(r.Signal == 1)[0], r.close[r.Signal == 1], marker='^', color='g', label='Buy Signal')
        ax.scatter(np.where(r.Signal == -1)[0], r.close[r.Signal == -1], marker='v', color='r', label='Sell Signal')
        ax.scatter(np.where(r.Signal == 10)[0], r.close[r.Signal == 10], marker='o', color='g', label='Buy Signal')
        ax.scatter(np.where(r.Signal == -10)[0], r.close[r.Signal == -10], marker='o', color='r', label='Sell Signal')
        for i, (x, y) in enumerate(zip(np.where(r.Signal != 0)[0], r.close[r.Signal != 0])):
            ax.text(x, y, f"{100 * (r.MACD[i + 1] - r.Signal_Line[i + 1]):.2f}({r.MACD[i + 1] * 100:.3f})", fontsize=7,
                    ha='right', va='bottom')
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()