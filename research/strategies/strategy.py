import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt


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
        shares_held = 0
        trades = 0
        positions = []  # Store updated signals
        if 'position' not in self.data.columns:
            self.data['position'] = self.data['signal']
        for index, row in self.data.iterrows():
            position = 0
            signal = row['signal']
            if shares_held > 0 and row['position'] == -1:
                # Sell signal
                position = -1
                trades += 1
                balance += row['close'] * shares_held
                print(f"Sold at: ${row['close']:.2f} x {shares_held}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f}")
                shares_held = 0

            elif balance > 0 and row['position'] == 1:
                # Buy signal
                shares_bought = balance // row['close']
                balance -= row['close'] * shares_bought
                shares_held += shares_bought
                if shares_bought:
                    print(f"share bought: {shares_bought:.2f}")
                    position = 1
                    print(f"Bought at: ${row['close']:.2f} x {shares_bought}  --- {row['macd_momentum']*100:.3f}")

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

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(np.arange(len(r)), r.close, linewidth=1)
        ax.scatter(np.where(r.signal == 1)[0], r.close[r.signal == 1], marker='^', color='g', label='Buy Signal')
        ax.scatter(np.where(r.signal == -1)[0], r.close[r.signal == -1], marker='v', color='r', label='Sell Signal')
        ax.scatter(np.where(r.position == 1)[0], r.close[r.position == 1], marker='o', color='g', alpha=.5, s=120, label='Buy')
        ax.scatter(np.where(r.position == -1)[0], r.close[r.position == -1], marker='o', color='r', alpha=.5, s=120, label='Sell')
        for i, (x, y) in enumerate(zip(np.where(r.signal != 0)[0], r.close[r.signal != 0])):
            if i < len(r.macd_momentum) - 1:
                ax.text(x, y, f"{r.macd_momentum[i+1] * 1000:.2f}", fontsize=7,
                        ha='right', va='bottom')
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
