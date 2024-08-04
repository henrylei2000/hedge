from strategy import Strategy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class WaveStrategy(Strategy):
    def wave_simple(self):
        short_window, long_window, signal_window = 12, 26, 9  # 3, 7, 2
        dataset = [self.data]
        if self.reference:
            dataset += [self.qqq, self.spy, self.dia]
        for data in dataset:
            # Calculate short-term and long-term exponential moving averages
            data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
            data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()

            # Calculate MACD line
            data['macd'] = data['short_ma'] - data['long_ma']
            # Calculate Signal line
            data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
            data['strength'] = data['macd'] - data['signal_line']

            price_change_ratio = data['close'].pct_change()
            data['vpt'] = (price_change_ratio * data['volume']).cumsum()
            data['rolling_vpt'] = data['vpt'].rolling(window=12).mean()

            data['obv'] = (data['volume'] * ((data['close'] - data['close'].shift(1)) > 0).astype(int) -
                           data['volume'] * ((data['close'] - data['close'].shift(1)) < 0).astype(int)).cumsum()
            # Calculate OBV moving average
            data['rolling_obv'] = data['obv'].rolling(window=12).mean()

            # Generate Buy and Sell signals
            data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

            data.to_csv(f"{self.symbol}.csv")

    def trend(self):
        self.wave_simple()
        data = self.data
        position = 0
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        hold = False
        count = 0
        for index, row in data.iterrows():
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            prominence = prices.iloc[0] * 0.00125 + 0.005
            print(f"----------- {prominence}")
            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=2, prominence=prominence)
            valleys, _ = find_peaks(-prices, distance=2, prominence=prominence)
            print(peaks)
            print(valleys)

            # Perform linear regression on peaks
            if len(peaks) > 5:
                peak_indices = np.array(peaks)
                peak_prices = prices.iloc[peaks]
                a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)
                a_recent, b_recent = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)
                if a_peaks * a_recent < 0:
                    if a_recent > a_peaks:
                        position = 1
                    print(
                        f"[{a_peaks:.3f} {a_recent:.3f}] [{b_peaks:.3f} {b_recent:.3f}] @{peak_indices[-1]} {data.index.get_loc(index)} {index}")

            # Perform linear regression on valleys
            if len(valleys) > 5:
                valley_indices = np.array(valleys)
                valley_prices = prices.iloc[valleys]
                a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)
                print(f"[{a_valleys:.3f} {b_valleys:.3f}]")

            if count == 90:
                self.snapshot(visible_rows, peaks, valleys)

            positions.append(position)
            count += 1
        data['position'] = positions

    def snapshot(self, rows, peaks, valleys):
        prices = rows['close']

        trends = pd.DataFrame({
            'Price': prices,
            'Trend': np.nan,
            'Type': np.nan
        }, dtype=object)

        # Use iloc to align indices correctly
        trends.iloc[peaks, trends.columns.get_loc('Trend')] = 'Peak'
        trends.iloc[valleys, trends.columns.get_loc('Trend')] = 'Valley'

        # Forward fill the trends to identify uptrends and downtrends
        trends['Trend'] = trends['Trend'].ffill()
        trends['Type'] = np.where(trends['Trend'] == 'Peak', 'Downtrend', 'Uptrend')

        # Perform linear regression on peaks
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)

        # Perform linear regression on valleys
        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]
        a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(prices, label='Price', color='blue')
        ax1.plot(prices.iloc[peaks], 'ro', label='Peaks')
        ax1.plot(prices.iloc[valleys], 'go', label='Valleys')
        ax1.plot(prices.index, a_peaks * np.arange(len(prices)) + b_peaks, 'r--', label='Peaks Linear Fit')
        ax1.plot(prices.index, a_valleys * np.arange(len(prices)) + b_valleys, 'g--', label='Valleys Linear Fit')
        ax1.set_title(f"{self.start} Stock Price Analysis")
        ax1.set_ylabel('Price')
        ax1.legend()

        ax2.plot(rows['macd'], label='OBV', color='purple')
        ax2.set_title('On-Balance Volume (OBV)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('OBV')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def signal(self):
        self.trend()
