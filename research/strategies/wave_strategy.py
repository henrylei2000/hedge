from strategy import Strategy
from scipy.signal import find_peaks
import numpy as np


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

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=1, prominence=0.1)
            valleys, _ = find_peaks(-prices, distance=1, prominence=0.1)

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

            positions.append(position)
            count += 1
        data['position'] = positions

    def signal(self):
        self.trend()