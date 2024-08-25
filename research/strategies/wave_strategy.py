from strategy import Strategy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, alternative_valley):
    # Convert lists to numpy arrays if they aren't already
    valley_indices = np.array(valley_indices)
    valley_prices = np.array(valley_prices)
    peak_indices = np.array(peak_indices)
    peak_prices = np.array(peak_prices)

    # Handle scenario 1: if the first peak appears before the first valley
    if peak_indices[0] < valley_indices[0]:
        valley_indices = np.insert(valley_indices, 0, 0)
        valley_prices = np.insert(valley_prices, 0, alternative_valley)

    # Create lists to store the corrected indices, peak indices, and valley indices
    corrected_indices = []
    corrected_peak_indices = []
    corrected_valley_indices = []

    # Start with the first valley
    i = j = 0
    while i < len(valley_indices) and j < len(peak_indices):
        # Handle consecutive valleys using argmin
        start_i = i
        while i + 1 < len(valley_indices) and valley_indices[i + 1] < peak_indices[j]:
            i += 1
        # Find the valley with the minimum price in the range
        min_price_idx = start_i + np.argmin(valley_prices[start_i:i + 1])
        corrected_indices.append(valley_indices[min_price_idx])
        corrected_valley_indices.append(valley_indices[min_price_idx])
        i += 1

        # Handle consecutive peaks using argmax
        start_j = j
        while j + 1 < len(peak_indices) and i < len(valley_indices) and peak_indices[j + 1] < valley_indices[i]:
            j += 1
        # Find the peak with the maximum price in the range
        max_price_idx = start_j + np.argmax(peak_prices[start_j:j + 1])
        corrected_indices.append(peak_indices[max_price_idx])
        corrected_peak_indices.append(peak_indices[max_price_idx])
        j += 1

    # Handle the case where only valleys are left
    if i < len(valley_indices):
        # Find the index with the minimum price among the remaining valleys
        min_price_idx = i + np.argmin(valley_prices[i:])
        corrected_indices.append(valley_indices[min_price_idx])
        corrected_valley_indices.append(valley_indices[min_price_idx])

    # Handle the case where only peaks are left
    if j < len(peak_indices):
        # Find the index with the maximum price among the remaining peaks
        max_price_idx = j + np.argmax(peak_prices[j:])
        corrected_indices.append(peak_indices[max_price_idx])
        corrected_peak_indices.append(peak_indices[max_price_idx])

    return np.array(corrected_indices), np.array(corrected_valley_indices), np.array(corrected_peak_indices)


# Example usage
valley_indices = [4, 7, 8, 9]
valley_prices = [105, 102, 89, 87]
peak_indices = [3, 5, 6]
peak_prices = [110, 108, 109]
prices = [98, 100, 108, 110, 105, 109, 108, 102, 89, 87]

prices = pd.Series(prices)

corrected_indices = rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, prices.iloc[0])
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

        corrected_indices, valley_indices, peak_indices = rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, prices.iloc[0])
        peak_prices = prices.iloc[peak_indices]
        valley_prices = prices.iloc[valley_indices]

        # Perform linear regression on peaks
        a_peaks, b_peaks = np.polyfit(peak_indices[-5:], peak_prices[-5:], 1)
        # Perform linear regression on valleys
        a_valleys, b_valleys = np.polyfit(valley_indices[-5:], valley_prices[-5:], 1)

        indicator = 'obv'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.1
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_peak_prices = obvs.iloc[obv_peaks]
        obv_valleys, _ = find_peaks(-obvs, distance=distance, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)
        obv_valley_prices = obvs.iloc[obv_valleys]

        obv_corrected_indices, obv_valley_indices, obv_peak_indices = rearrange_valley_peak(obv_valley_indices, obv_valley_prices, obv_peak_indices, obv_peak_prices, obvs.iloc[0])
        obv_peak_prices = obvs.iloc[obv_peak_indices]
        obv_valley_prices = obvs.iloc[obv_valley_indices]

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
        ax1.plot(prices.iloc[peak_indices], 'ro', label='Peaks')
        for peak in peak_indices:
            ax1.annotate(f'{peak}',
                         (prices.index[peak], prices.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax1.plot(prices.iloc[valley_indices], 'go', label='Valleys')
        for valley in valley_indices:
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
        ax2.plot(obvs.iloc[obv_peak_indices], 'ro', label='Peaks')
        print(f"----------------- {len(obv_peaks)}")
        # Annotate each peak with its value
        for peak in obv_peak_indices:
            ax2.annotate(f'{peak}',
                         (obvs.index[peak], obvs.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax2.plot(obvs.iloc[obv_valley_indices], 'go', label='Valleys')
        for valley in obv_valley_indices:
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
        prominence = data.iloc[0]['close'] * 0.00169 + 0.003
        obv_prominence = data.iloc[0]['obv'] * 0.1
        macd_prominence = data.iloc[0]['obv'] * 0.1

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            if len(peak_indices) and len(valley_indices):
                corrected_indices, valley_indices, peak_indices = rearrange_valley_peak(valley_indices, valley_prices,
                                                                                        peak_indices, peak_prices,
                                                                                        prices.iloc[0])
                peak_prices = prices.iloc[peak_indices]
                valley_prices = prices.iloc[valley_indices]

            obvs = visible_rows['obv']
            # Identify peaks and valleys
            obv_peaks, _ = find_peaks(obvs, distance=distance, prominence=obv_prominence)
            obv_peak_indices = np.array(obv_peaks)
            obv_peak_prices = obvs.iloc[obv_peaks]
            obv_valleys, _ = find_peaks(-obvs, distance=distance, prominence=obv_prominence)
            obv_valley_indices = np.array(obv_valleys)
            obv_valley_prices = obvs.iloc[obv_valleys]

            if len(obv_valley_indices) and len(obv_peak_indices):
                obv_corrected_indices, obv_valley_indices, obv_peak_indices = rearrange_valley_peak(obv_valley_indices,
                                                                                                    obv_valley_prices,
                                                                                                    obv_peak_indices,
                                                                                                    obv_peak_prices,
                                                                                                    obvs.iloc[0])
                obv_peak_prices = obvs.iloc[obv_peak_indices]
                obv_valley_prices = obvs.iloc[obv_valley_indices]

            if len(valley_indices) and (len(peak_indices) and valley_indices[-1] > peak_indices[-1] or not len(peak_indices)):
                print(f"------- uphill span {count - valley_indices[-1]} up {price - prices.iloc[valley_indices[-1]]}")
            if len(peak_indices) and (len(valley_indices) and peak_indices[-1] > valley_indices[-1] or not len(valley_indices)):
                print(f"------- downhill span {count - peak_indices[-1]} down {prices.iloc[peak_indices[-1]] - price}")

            if len(obv_valleys) > obv_num_valleys:
                print(f"Found a new obv valley after {count - obv_valleys[-1]}")
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
