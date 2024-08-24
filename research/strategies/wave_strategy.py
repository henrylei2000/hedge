from strategy import Strategy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, prices):
    # Handle scenario 1: if the first peak appears before the first valley
    if peak_indices[0] < valley_indices[0]:
        valley_indices.insert(0, 0)
        valley_prices.insert(0, prices[0])

    # Create a new list to store the corrected indices and prices
    corrected_indices = []
    corrected_prices = []

    # Start with the first valley
    i = j = 0
    while i < len(valley_indices) and j < len(peak_indices):
        # Add the valley
        corrected_indices.append(valley_indices[i])
        corrected_prices.append(valley_prices[i])
        i += 1

        # Check for consecutive peaks
        while j + 1 < len(peak_indices) and peak_indices[j + 1] < valley_indices[i]:
            if peak_prices[j + 1] > peak_prices[j]:
                j += 1
            else:
                peak_indices.pop(j + 1)
                peak_prices.pop(j + 1)
        corrected_indices.append(peak_indices[j])
        corrected_prices.append(peak_prices[j])
        j += 1

    # Handle the case where only valleys are left
    if i < len(valley_indices):
        # Find the index with the minimum price among the remaining valleys
        min_price_idx = i + valley_prices[i:].index(min(valley_prices[i:]))
        corrected_indices.append(valley_indices[min_price_idx])
        corrected_prices.append(valley_prices[min_price_idx])

    # Handle the case where only peaks are left
    if j < len(peak_indices):
        # Find the index with the maximum price among the remaining peaks
        max_price_idx = j + peak_prices[j:].index(max(peak_prices[j:]))
        corrected_indices.append(peak_indices[max_price_idx])
        corrected_prices.append(peak_prices[max_price_idx])

    return corrected_indices

# Example usage
valley_indices = [4, 7, 8]
valley_prices = [105, 102, 89]
peak_indices = [3, 5, 6]
peak_prices = [110, 109, 108]
prices = [98, 100, 108, 110, 105, 109, 108, 102, 89]

corrected_indices = rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, prices)
print("Corrected Indices:", corrected_indices)


def standout(values):
    base = values.iloc[-1]
    high, low = 0, 0
    high_stop, low_stop = False, False
    for v in values[-2::-1]:
        if v <= base and not high_stop:
            high += 1
            low_stop = True
        elif not low_stop:
            low += 1
            high_stop = True
    return high, low


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

    def snapshot(self, interval, distance, prominence):
        if interval[1] - interval[0] < 30:
            return
        rows = self.data.iloc[interval[0]:interval[1]]
        prices = rows['close']

        # Identify peaks and valleys
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]

        # Perform linear regression on peaks
        a_peaks, b_peaks = np.polyfit(peak_indices[-5:], peak_prices[-5:], 1)
        # Perform linear regression on valleys
        a_valleys, b_valleys = np.polyfit(valley_indices[-5:], valley_prices[-5:], 1)

        indicator = 'obv'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.1
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance * 2, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_peak_prices = obvs.iloc[obv_peaks]
        obv_valleys, _ = find_peaks(-obvs, distance=distance * 2, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)
        obv_valley_prices = obvs.iloc[obv_valleys]

        # Perform linear regression on peaks
        obv_a_peaks, obv_b_peaks = np.polyfit(obv_peak_indices[-5:], obv_peak_prices[-5:], 1)
        # Perform linear regression on valleys
        obv_a_valleys, obv_b_valleys = np.polyfit(obv_valley_indices[-5:], obv_valley_prices[-5:], 1)

        # Get positions for buy (1) and sell (-1) signals
        buy_signals = rows[rows['position'] == 1]
        sell_signals = rows[rows['position'] == -1]
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        ax1.plot(prices, label='Price', color='blue')
        ax1.plot(prices.iloc[peaks], 'ro', label='Peaks')
        for peak in peaks:
            ax1.annotate(f'{peak}',
                         (prices.index[peak], prices.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax1.plot(prices.iloc[valleys], 'go', label='Valleys')
        for valley in valleys:
            ax1.annotate(f'{valley}',
                         (prices.index[valley], prices.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax1.plot(prices.index, a_peaks * np.arange(len(prices)) + b_peaks, 'r--', label='Peaks Linear Fit')
        ax1.plot(prices.index, a_valleys * np.arange(len(prices)) + b_valleys, 'g--', label='Valleys Linear Fit')
        # Plot buy and sell signals
        ax1.plot(buy_signals.index, buy_signals['close'], 'g^', markersize=12, alpha=.5, label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['close'], 'rv', markersize=12, alpha=.5, label='Sell Signal')
        ax1.set_title(f"{self.start.strftime('%Y-%m-%d')} {interval} Stock Price Analysis")
        ax1.set_ylabel('Price')
        ax1.legend()

        ax2.plot(rows[indicator], label=f"{indicator}", color='purple')
        ax2.plot(obvs.iloc[obv_peaks], 'ro', label='Peaks')
        # Annotate each peak with its value
        for peak in obv_peaks:
            ax2.annotate(f'{peak}',
                         (obvs.index[peak], obvs.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax2.plot(obvs.iloc[obv_valleys], 'go', label='Valleys')
        for valley in obv_valleys:
            ax2.annotate(f'{valley}',
                         (obvs.index[valley], obvs.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax2.plot(obvs.index, obv_a_peaks * np.arange(len(obvs)) + obv_b_peaks, 'r--', label='Peaks Linear Fit')
        ax2.plot(obvs.index, obv_a_valleys * np.arange(len(obvs)) + obv_b_valleys, 'g--', label='Valleys Linear Fit')
        ax2.set_title(f"{indicator}")
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f"{indicator}")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def trend(self):
        self.wave_simple()
        data = self.data
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        obv_bullish, macd_bullish, price_bullish, hold = False, False, False, False
        a_peaks = 1000000
        b_peaks = 1000000
        sell_point = 0
        count = 0
        num_peaks, num_valleys = 0, 0
        obv_num_peaks, obv_num_valleys = 0, 0
        macd_num_peaks, macd_num_valleys = 0, 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005
        obv_prominence = data.iloc[0]['obv'] * 0.1
        macd_prominence = data.iloc[0]['obv'] * 0.1

        for index, row in data.iterrows():
            position = 0
            print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {row['close']:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            obvs = visible_rows['obv']
            # Identify peaks and valleys
            obv_peaks, _ = find_peaks(obvs, distance=distance, prominence=obv_prominence)
            obv_peak_indices = np.array(obv_peaks)
            obv_peak_prices = obvs.iloc[obv_peaks]
            obv_valleys, _ = find_peaks(-obvs, distance=distance, prominence=obv_prominence)
            obv_valley_indices = np.array(obv_valleys)
            obv_valley_prices = obvs.iloc[obv_valleys]

            if len(obv_valleys) > obv_num_valleys:
                print(f"Found a new obv valley after {count - obv_valleys[-1]}")
                """
                wave_measure: span, gap, wave_num_max
                """
                print(
                    f"OBV Valley standout: {standout(obv_valley_prices)}, recent obv valleys {obv_valley_indices[-3:]}")
                obv_num_valleys += 1
                if obv_num_valleys > 1 and standout(obv_valley_prices)[0] > standout(obv_valley_prices[:-1])[
                    0] == 0:
                    print(f"an obv reversal @ {obv_valleys[-1]}")
                    obv_bullish = True

            if count and row['strength'] > 0 > data.iloc[count-1]['strength']:
                macd_bullish = True

            if len(valleys) > num_valleys:
                print(f"Found a new valley after {count - valleys[-1]}")
                print(f"Valley standout: {standout(valley_prices)}, recent valleys {valley_indices[-3:]}")
                num_valleys += 1
                if num_valleys and (num_peaks and valleys[-1] > peaks[-1] or not num_peaks):
                    print(f"trending up from the recent valley")
                    price_bullish = True

            if len(obv_peaks) > obv_num_peaks:  # new peak found!
                print(f"Found a new obv peak after {count - obv_peaks[-1]}")
                print(f"OBV Peak standout: {standout(obv_peak_prices)}, recent obv peaks {obv_peak_indices[-3:]}")
                obv_num_peaks += 1
                obv_bullish = False

            if count and row['strength'] < 0 < data.iloc[count-1]['strength']:
                macd_bullish = False

            if len(peaks) > num_peaks:  # new peak found!
                print(f"Found a new peak after {count - peaks[-1]}")
                print(f"Peak standout: {standout(peak_prices)}")
                num_peaks += 1
                if num_peaks and (num_valleys and valleys[-1] < peaks[-1] or not num_valleys):
                    print(f"trending down from the recent peak")
                    price_bullish = False

            if num_peaks > 1:
                # Perform linear regression on peaks
                a_peaks, b_peaks = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)

            list_price = a_peaks * count + b_peaks

            if num_valleys > 1:
                # Perform linear regression on valleys
                a_valleys, b_valleys = np.polyfit(valley_indices[-5:], valley_prices[-5:], 1)

                uphill = peak_prices[-1] - valley_prices[-2]
                downhill = peak_prices[-1] - valley_prices[-1]

                print(f"---- uphill {uphill}, downhill {downhill}")

            if price_bullish and obv_bullish and not hold:
                if row['close'] > list_price:
                    position = 1
                    hold = True
                    print(f"buying @{count} {row['close']}")

            if not macd_bullish and not price_bullish and not obv_bullish and hold:
                position = -1
                hold = False
                print(f"selling @{count} {row['close']}")

            positions.append(position)
            count += 1
            print("\n")

        data['position'] = positions
        self.snapshot([0, 130], distance, prominence)

    def signal(self):
        self.trend()
