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
        buy, hold = False, False
        count = 0
        num_peaks, num_valleys = 0, 0
        bottom, bottom_index = 0, 0

        prominence = data.iloc[0]['close'] * 0.00125 + 0.005
        print(f"prominence ----------- {prominence}")

        for index, row in data.iterrows():
            print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {row['close']:.4f} / {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=2, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=2, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            if len(peaks) > num_peaks:  # new peak found!
                print(f"Found a new peak after {count - peaks[-1]}")
                num_peaks += 1
            if len(valleys) > num_valleys:
                print(f"Found a new valley after {count - valleys[-1]}")
                num_valleys += 1
                if min(valley_prices) == valley_prices.iloc[-1]:  # lowest valley
                    bottom = valley_prices.iloc[-1]
                    bottom_index = valley_indices[-1]
                    print(f"[Trending HIGH] valley is the lowest: {bottom} {bottom_index}")
                elif min(valley_prices) < valley_prices.iloc[-1]:
                    print(f"Buy signal @ {count}")
                    if not hold:
                        buy = True


            # Perform linear regression on peaks
            if num_peaks > 2:
                a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)
                projected_peak = a_peaks * count + b_peaks
                a_recent, b_recent = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)
                projected_recent = a_recent * count + b_recent
                print(f"project peak {projected_peak:.4f} and {projected_recent:.4f}")

                if a_peaks * a_recent < 0:  # trend reversal
                    if a_recent < a_peaks:
                        position = -1
                        print(f"xxxxxxxxxxxxxxxxxxxxxxx")
                    print(
                        f"[{a_peaks:.3f} {a_recent:.3f}] [{b_peaks:.3f} {b_recent:.3f}] @{peak_indices[-1]}")

                if max(peak_prices) == peak_prices.iloc[-1]:  # highest peak
                    print(
                        f"[Trending LOW!] peak is the highest: {peak_prices.iloc[-1]} {visible_rows.iloc[peak_indices[-1]]['close']}")

            # Perform linear regression on valleys
            if num_valleys > 2:
                a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)
                projected_valley = a_valleys * count + b_valleys

                if buy:
                    print(f"---------- {bottom_index}")
                    recent_indices = [i for i in valleys if i >= bottom_index]
                    print(f"recent indices {recent_indices}")
                    recent_prices = prices.iloc[recent_indices]
                    a_recent, b_recent = np.polyfit(recent_indices, recent_prices, 1)

                    projected_recent = a_recent * count + b_recent
                    print(f"project valley {projected_valley:.4f} and {projected_recent:.4f}")
                    if a_valleys * a_recent < 0:  # trend reversal
                        if a_recent > a_valleys:
                            position = 1
                            print(f"!!!!!!!!!!!!!!!!!")
                        print(
                            f"[{a_valleys:.3f} {a_recent:.3f}] [{b_valleys:.3f} {b_recent:.3f}] @{valley_indices[-1]}")

            if count == 20:
                print(f"last dip @{bottom_index} {bottom} Strength diff: {visible_rows.iloc[bottom_index]['strength']} {visible_rows.iloc[bottom_index + 1]['strength']} {row['strength']}")
                self.snapshot(visible_rows, peaks, valleys)

            positions.append(position)
            count += 1
            print("\n")

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
        ax1.set_title(f"{self.start.strftime('%Y-%m-%d')} Stock Price Analysis")
        ax1.set_ylabel('Price')
        ax1.legend()

        label = 'obv'
        ax2.plot(rows[label], label=f"{label}", color='purple')
        ax2.set_title(f"{label}")
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f"{label}")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def signal(self):
        self.trend()
