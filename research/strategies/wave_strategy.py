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


    def standout(self, values):
        base = values[-1]
        count = 0
        for v in values[-2::-1]:
            if v <= base:
                count += 1
            else:
                break

        return count

    def snapshot(self, interval, prominence):

        rows = self.data.iloc[interval[0]:interval[1]]
        prices = rows['close']

        # Identify peaks and valleys
        peaks, _ = find_peaks(prices, distance=2, prominence=prominence)
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        valleys, _ = find_peaks(-prices, distance=2, prominence=prominence)
        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]

        # Perform linear regression on peaks
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        a_peaks, b_peaks = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)

        # Perform linear regression on valleys
        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]
        a_valleys, b_valleys = np.polyfit(valley_indices[-3:], valley_prices[-3:], 1)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(prices, label='Price', color='blue')
        ax1.plot(prices.iloc[peaks], 'ro', label='Peaks')
        ax1.plot(prices.iloc[valleys], 'go', label='Valleys')
        ax1.plot(prices.index, a_peaks * np.arange(len(prices)) + b_peaks, 'r--', label='Peaks Linear Fit')
        ax1.plot(prices.index, a_valleys * np.arange(len(prices)) + b_valleys, 'g--', label='Valleys Linear Fit')
        ax1.set_title(f"{self.start.strftime('%Y-%m-%d')} {interval} Stock Price Analysis")
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

    def trend(self):
        self.wave_simple()
        data = self.data
        position = 0
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        buy, sell, hold = False, False, False
        sell_point = 0
        count = 0
        num_peaks, num_valleys = 0, 0
        bottom, bottom_index = 0, 0
        projected_peak = 0
        a_valleys, b_valleys = 0, 0
        prominence = data.iloc[0]['close'] * 0.00168 + 0.005
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
                print(f"Peak standout: {self.standout(peak_prices)}")
                num_peaks += 1
                buy = False
                if hold and peak_prices.iloc[-1] < peak_prices.iloc[-2]:
                    print(f"Peak not reached {count} {peak_prices.iloc[-1]} {a_valleys * peaks[-1] + b_valleys}")
                    sell = True
                if max(peak_prices) == peak_prices.iloc[-1]:  # highest peak
                    print(
                        f"[Trending LOW!] peak is the highest: {peak_prices.iloc[-1]} {visible_rows.iloc[peak_indices[-1]]['close']}")

            if len(valleys) > num_valleys:
                print(f"Found a new valley after {count - valleys[-1]}")
                print(f"Valley standout: {self.standout(valley_prices)}")
                num_valleys += 1
                if hold:
                    sell_point = max(sell_point, valley_prices.iloc[-1])
                if min(valley_prices) == valley_prices.iloc[-1]:  # lowest valley
                    bottom = valley_prices.iloc[-1]
                    bottom_index = valley_indices[-1]
                    print(f"[Trending HIGH] valley is the lowest: {bottom} {bottom_index}")

                if self.standout(valley_prices) > 2 and self.standout(peak_prices) > 2 and self.standout(valley_prices[:-1]) and self.standout(peak_prices[:-1]):

                    if not hold:
                        print(f"Buy signal @ {count}")
                        buy = True

            # Perform linear regression on valleys
            if num_valleys > 2 and num_peaks > 2:
                a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)
                projected_peak = a_peaks * count + b_peaks
                a_recent, b_recent = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)
                projected_recent = a_recent * count + b_recent
                print(f"project peak {projected_peak:.4f} and {projected_recent:.4f}")

                if row['close'] <= sell_point and sell:  # trend reversal
                    position = -1
                    hold = False
                    sell = False
                    sell_point = 0
                    print(f"selling @ {row['close']}")
                    print(
                        f"[{a_peaks:.3f} {a_recent:.3f}] [{b_peaks:.3f} {b_recent:.3f}] @{peak_indices[-1]}")

                a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)
                projected_valley = a_valleys * count + b_valleys

                if buy:
                    print(f"Buy signal since the last dip ---------- {bottom_index}")
                    recent_indices = [i for i in valleys if i >= bottom_index]
                    print(f"recent indices {recent_indices}")
                    recent_prices = prices.iloc[recent_indices]
                    a_recent, b_recent = np.polyfit(recent_indices, recent_prices, 1)

                    projected_recent = a_recent * count + b_recent
                    print(f"project valley {projected_valley:.4f} and {projected_recent:.4f}")

                    position = 1
                    print(f"buying @ {row['close']}")
                    buy = False
                    hold = True
                    sell_point = row['close']
                    print(
                            f"[{a_valleys:.3f} {a_recent:.3f}] [{b_valleys:.3f} {b_recent:.3f}] @{valley_indices[-1]}")

                print(f"last dip @{bottom_index} {bottom} Strength diff: {visible_rows.iloc[bottom_index]['strength']} {visible_rows.iloc[bottom_index + 1]['strength']} {row['strength']}")
            positions.append(position)
            count += 1
            print("\n")

        data['position'] = positions
        self.snapshot([300, 385], prominence)

    def signal(self):
        self.trend()
