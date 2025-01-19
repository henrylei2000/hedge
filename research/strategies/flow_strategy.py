from strategy import Strategy
from collections import deque
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def calculate_recent_macd_divergence_with_crossover(price_data, macd_data, divergence_window=5):
    recent_prices = price_data[-divergence_window:]
    recent_macd = macd_data[-divergence_window:]

    macd_divergence = None

    # Bullish MACD Divergence: Price makes lower lows, MACD makes higher lows
    if min(recent_prices) == recent_prices.iloc[-1] and min(recent_macd) < recent_macd.iloc[0]:
        macd_divergence = 'bullish'

    # Bearish MACD Divergence: Price makes higher highs, MACD makes lower highs
    elif max(recent_prices) == recent_prices.iloc[-1] and max(recent_macd) > recent_macd.iloc[0]:
        macd_divergence = 'bearish'

    return macd_divergence


def detect_recent_ad_line_divergence(prices, ads, divergence_window=5):
    recent_prices = prices[-divergence_window:]
    recent_ads = ads[-divergence_window:]

    if min(recent_prices) == recent_prices.iloc[-1] and min(recent_ads) < recent_ads.iloc[0]:
        return 'buy'
    elif max(recent_prices) == recent_prices.iloc[-1] and max(recent_ads) > recent_ads.iloc[0]:
        return 'sell'
    else:
        return None


def calculate_recent_fibonacci_levels(price_data, peaks, valleys, divergence_window=5):
    most_recent_peak = peaks[peaks < len(price_data)].max() if len(peaks) > 0 else None
    most_recent_valley = valleys[valleys < len(price_data)].max() if len(valleys) > 0 else None
    recent_fib_levels = None

    if most_recent_valley is not None and most_recent_valley > most_recent_peak:
        high = price_data.iloc[most_recent_peak]
        low = price_data.iloc[most_recent_valley]
        recent_fib_levels = {f"{int(ratio * 100)}%": high - (high - low) * ratio for ratio in
                             [0.236, 0.382, 0.5, 0.618, 1.0]}

    elif most_recent_peak is not None and most_recent_valley is not None and most_recent_peak > most_recent_valley:
        high = price_data.iloc[most_recent_peak]
        low = price_data.iloc[most_recent_valley]
        recent_fib_levels = {f"{int(ratio * 100)}%": low + (high - low) * ratio for ratio in
                             [1.0, 1.272, 1.618, 2.0, 2.618]}

    return recent_fib_levels


def rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, alternative_valley):
    # Convert lists to numpy arrays if they aren't already
    valley_indices = np.array(valley_indices)
    valley_prices = np.array(valley_prices)
    peak_indices = np.array(peak_indices)
    peak_prices = np.array(peak_prices)

    # Handle scenario 1: if the first peak appears before the first valley
    if len(peak_indices) and len(valley_indices) and peak_indices[0] < valley_indices[0]:
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


def ad_line(prices, high, low, volume):
    # Money Flow Multiplier calculation
    money_flow_multiplier = ((prices - low) - (high - prices)) / (high - low)
    # Money Flow Volume calculation
    money_flow_volume = money_flow_multiplier * volume
    # Accumulation/Distribution Line (cumulative sum of money flow volume)
    ad_line = money_flow_volume.cumsum()
    return ad_line


def adaptive_macd(prices, turning_points):
    wave_lengths = [turning_points[i] - turning_points[i - 1] for i in range(1, len(turning_points))]
    signal_window = sum(wave_lengths[-2:]) if len(wave_lengths) >= 2 else 9
    long_window = sum(wave_lengths[-5:]) if len(wave_lengths) >= 5 else 26
    short_window = sum(wave_lengths[-3:]) if len(wave_lengths) >= 3 else 12
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line


def wave_macd(prices):
    short_window, long_window, signal_window = 2, 3, 1
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line


def standout(values):
    base = values.iloc[-1]
    high, low = 0, 0
    high_stop, low_stop = False, False
    for v in values[-2::-1]:
        if v <= base:
            low_stop = True
            if not high_stop:
                high += 1
        elif v > base:
            high_stop = True
            if not low_stop:
                low += 1

    return high, low


class FlowStrategy(Strategy):
    def flow_simple(self):
        short_window, long_window, signal_window = 12, 26, 9   # 9, 21, 6
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

            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=signal_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=signal_window).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))

            price_change_ratio = data['close'].pct_change()
            data['vpt'] = (price_change_ratio * data['volume']).cumsum()
            data['rolling_vpt'] = data['vpt'].rolling(window=12).mean()

            data['obv'] = (data['volume'] * ((data['close'] - data['close'].shift(1)) > 0).astype(int) -
                           data['volume'] * ((data['close'] - data['close'].shift(1)) < 0).astype(int)).cumsum()
            # Calculate OBV moving average
            data['rolling_obv'] = data['obv'].rolling(window=12).mean()
            data['rolling_volume'] = data['obv'].rolling(window=3).mean()
            data['a/d'] = ad_line(data['close'], data['high'], data['low'], data['volume'])
            # Generate Buy and Sell signals
            data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
            # data.to_csv(f"{self.symbol}.csv")

    def zero_crossing(self):
        self.flow_simple()
        data = self.data
        previous = deque(maxlen=3)  # Keep track of the last 3 signals
        prev_peak = deque(maxlen=3)
        prev_valley = deque(maxlen=3)
        positions = []  # Store updated signals
        hold = False
        # Initialize Signal column with zeros
        data['position'] = 0
        distance = 9
        prominence = data.iloc[0]['close'] * 0.002

        for index, row in data.iterrows():
            position = 0
            current, current_peak, current_valley = 0, 0, 0
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']

            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            if len(peak_indices) > 1 and len(valley_indices):
                corrected_indices, valley_indices, peak_indices = rearrange_valley_peak(valley_indices, valley_prices,
                                                                                        peak_indices, peak_prices,
                                                                                        prices.iloc[0])

                macd_peak, signal_peak = adaptive_macd(prices, peak_indices)
                macd_valley, signal_valley = adaptive_macd(prices, valley_indices)
                macd = macd_peak.iloc[-1] + macd_valley.iloc[-1]
                signal = signal_peak.iloc[-1] + signal_valley.iloc[-1]

                macd_peak, signal_peak = wave_macd(peak_prices)
                macd_valley, signal_valley = wave_macd(valley_prices)

                current_peak = macd_peak.iloc[-1]  # - signal_peak.iloc[-1]
                current_valley = macd_valley.iloc[-1]  # - signal_valley.iloc[-1]
                # current = row['strength']
                if len(prev_peak):
                    if prev_peak[-1] > 0 > current_peak and hold:
                        position = -1
                        hold = False
                if len(prev_valley):
                    if prev_valley[-1] < 0 < current_valley and not hold:
                        position = 1
                        hold = True

            positions.append(position)
            previous.append(current)
            prev_peak.append(current_peak)
            prev_valley.append(current_valley)

        data['position'] = positions
        self.snapshot([0, 89], distance, prominence)

    def macd_divergence(self, divergence_window=15, crossover_confirmation=False):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        count = 0

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            # print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            macds = visible_rows['macd']
            strengths = visible_rows['strength']
            adlines = visible_rows['a/d']

            if len(prices) < divergence_window or len(macds) < divergence_window:
                positions.append(position)
                count += 1
                continue

            # Get the most recent window of prices and MACD values
            recent_prices = prices[-divergence_window:]
            recent_macd = macds[-divergence_window:]

            # Bullish MACD Divergence: Price makes lower lows, MACD makes higher lows
            if min(recent_prices) == recent_prices.iloc[-1] and min(recent_macd) < recent_macd.iloc[0]:
                position = 1
                hold = True

           # Bearish MACD Divergence: Price makes higher highs, MACD makes lower highs
            elif max(recent_prices) == recent_prices.iloc[-1] and max(recent_macd) > recent_macd.iloc[0]:
                position = -1
                hold = False
            else:
                position = 0

            positions.append(position)
            count += 1

        data['position'] = positions

    def ad_divergence(self, divergence_window=5):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        count = 0

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            # print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            ads = visible_rows['a/d']

            if len(prices) < divergence_window or len(ads) < divergence_window:
                positions.append(position)
                count += 1
                continue

            # Get the most recent window of prices and MACD values
            recent_prices = prices[-divergence_window:]
            recent_ads = ads[-divergence_window:]

            # Bullish MACD Divergence: Price makes lower lows, MACD makes higher lows
            if min(recent_prices) == recent_prices.iloc[-1] and min(recent_ads) < recent_ads.iloc[0]:
                position = 1
                hold = True

            # Bearish MACD Divergence: Price makes higher highs, MACD makes lower highs
            elif max(recent_prices) == recent_prices.iloc[-1] and max(recent_ads) > recent_ads.iloc[0]:
                position = -1
                hold = False
            else:
                position = 0

            positions.append(position)
            count += 1

        data['position'] = positions

    def flow(self, divergence_window = 5):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        count = 0

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            # print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            macds = visible_rows['macd']
            ads = visible_rows['a/d']

            macd_divergence = calculate_recent_macd_divergence_with_crossover(prices, macds, divergence_window)
            ad_divergence = detect_recent_ad_line_divergence(prices, ads, divergence_window)

            peaks, _ = find_peaks(prices, distance=divergence_window)
            valleys, _ = find_peaks(-prices, distance=divergence_window)
            if len(peaks) and len(valleys):
                fib_levels = calculate_recent_fibonacci_levels(prices, peaks, valleys, divergence_window)

                # Generate buy/sell signals based on combined logic
                if macd_divergence == 'bullish' and ad_divergence == 'buy' and fib_levels:
                    if '61%' in fib_levels and prices.iloc[-1] >= fib_levels['61%'] or True:
                        position = 1
                        hold = True

                elif macd_divergence == 'bearish' and ad_divergence == 'sell' and fib_levels:
                    if '161%' in fib_levels and prices.iloc[-1] <= fib_levels['161%'] or True:
                        position = -1
                        hold = False

            positions.append(position)
            count += 1

        data['position'] = positions

    def deep_v(self, lookback=5, lookforward=3, decline_threshold=0.005, recovery_threshold=0.003):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        count = 0

        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            volumes = visible_rows['volume']
            macds = visible_rows['macd']
            rsis = visible_rows['rsi']
            ads = visible_rows['a/d']

            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            if len(peak_indices) > 1 and len(valley_indices):
                corrected_indices, valley_indices, peak_indices = rearrange_valley_peak(valley_indices, valley_prices,
                                                                                        peak_indices, peak_prices,
                                                                                        prices.iloc[0])
                valley, peak = valley_indices[-1], peak_indices[-1]
                if valley > peak and len(prices) < valley + lookforward:  # from a valley
                    deep_v = 0

                    pre_volume = volumes.iloc[max(0, valley - lookback):valley].mean()
                    post_volume = volumes.iloc[valley + 1:valley + lookforward].max()
                    if post_volume > pre_volume * 1.5:  # At least 1.5x average volume
                        deep_v += 1

                    # Check Volume Spike
                    pre_macd = macds.iloc[max(0, valley - lookback):valley].mean()
                    post_macd = macds.iloc[valley + 1:valley + lookforward].mean()
                    if post_macd > pre_macd:  # At least 1.5x average volume
                        deep_v += 1

                    if rsis.iloc[valley] < 30:
                        deep_v += 1

                    pre_valley_price = prices.iloc[valley - lookback:valley].max()
                    pre_decline = (pre_valley_price - prices.iloc[valley]) / pre_valley_price
                    post_valley_price = prices.iloc[valley + 1:valley + 1 + lookforward].max()
                    post_recovery = (post_valley_price - prices.iloc[valley]) / prices.iloc[valley]
                    # Check thresholds for both decline and recovery
                    if pre_decline >= decline_threshold and post_recovery >= recovery_threshold:
                        deep_v += 1
                    else:
                        deep_v = 0

                    if deep_v > 0:
                        position = deep_v
                        print(f"---- deep_v scored {deep_v} @ {count}")
                        hold = True

                if peak > valley:
                    if hold and rsis.iloc[peak] > 70:
                        position = -4
                        hold = False

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([180, 249], distance, prominence)

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

        indicator = 'rsi'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.00125 + 0.005
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance*3, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_peak_prices = obvs.iloc[obv_peaks]
        obv_valleys, _ = find_peaks(-obvs, distance=distance*3, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)
        obv_valley_prices = obvs.iloc[obv_valleys]

        obv_corrected_indices, obv_valley_indices, obv_peak_indices = rearrange_valley_peak(obv_valley_indices, obv_valley_prices, obv_peak_indices, obv_peak_prices, obvs.iloc[0])
        obv_peak_prices = obvs.iloc[obv_peak_indices]
        obv_valley_prices = obvs.iloc[obv_valley_indices]

        # debugging info
        dips = []
        for i in range(valley_indices.size - 1):
            if valley_indices[i] > valley_indices[-1] - 60:
                dips.append((valley_indices[i], valley_indices[-1]))

        print(f"[SNAPSHOT] {dips}")

        # Get positions for buy (1) and sell (-1) signals
        buy_signals = rows[rows['position'] > 0]
        sell_signals = rows[rows['position'] < 0]
        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})

        ax1.plot(prices, label='Price', color='blue')
        ax1.plot(prices.iloc[peak_indices], 'ro', label='Peaks')
        for peak in peak_indices:
            ax1.annotate(f'{interval[0] + peak}',
                         (prices.index[peak], prices.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax1.plot(prices.iloc[valley_indices], 'go', label='Valleys')
        for valley in valley_indices:
            ax1.annotate(f'{interval[0] + valley}',
                         (prices.index[valley], prices.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        # Plot buy and sell signals
        ax1.plot(buy_signals.index, buy_signals['close'], 'g^', markersize=12, alpha=.5, label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['close'], 'rv', markersize=12, alpha=.5, label='Sell Signal')
        ax1.set_title(f"{self.start.strftime('%Y-%m-%d')} {interval} Stock Price Analysis")
        ax1.set_ylabel('Price')
        ax1.legend()

        ax2.plot(rows[indicator], label=f"{indicator}", color='lightblue')
        ax2.plot(obvs.iloc[obv_peak_indices], 'ro', label='peaks')
        # Annotate each peak with its value
        for peak in obv_peak_indices:
            ax2.annotate(f'{interval[0] + peak}',
                         (obvs.index[peak], obvs.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax2.plot(obvs.iloc[obv_valley_indices], 'go', label='valleys')
        for valley in obv_valley_indices:
            ax2.annotate(f'{interval[0] + valley}',
                         (obvs.index[valley], obvs.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        # if len(obv_peak_indices):
        #     ax2.plot(obvs.index, obv_a_peaks * np.arange(len(obvs)) + obv_b_peaks, 'r--', label='Peaks Linear Fit')
        #     ax2.plot(obvs.index, obv_a_valleys * np.arange(len(obvs)) + obv_b_valleys, 'g--', label='Valleys Linear Fit')
        ax2.legend()

        indicator = 'a/d'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.00125 + 0.005
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_peak_prices = obvs.iloc[obv_peaks]
        obv_valleys, _ = find_peaks(-obvs, distance=distance, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)
        obv_valley_prices = obvs.iloc[obv_valleys]

        obv_corrected_indices, obv_valley_indices, obv_peak_indices = rearrange_valley_peak(obv_valley_indices,
                                                                                            obv_valley_prices,
                                                                                            obv_peak_indices,
                                                                                            obv_peak_prices,
                                                                                            obvs.iloc[0])
        obv_peak_prices = obvs.iloc[obv_peak_indices]
        obv_valley_prices = obvs.iloc[obv_valley_indices]
        ax3.plot(rows[indicator], label=f"{indicator}", color='pink')
        ax3.plot(obvs.iloc[obv_peak_indices], 'ro', label='peaks')
        # Annotate each peak with its value
        for peak in obv_peak_indices:
            ax3.annotate(f'{interval[0] + peak}',
                         (obvs.index[peak], obvs.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax3.plot(obvs.iloc[obv_valley_indices], 'go', label='valleys')
        for valley in obv_valley_indices:
            ax3.annotate(f'{interval[0] + valley}',
                         (obvs.index[valley], obvs.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def trend(self):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        wavelength, wavestart, entry, patience, next_peak = 0, 0, 0, 0, False
        buy_price = 0
        count = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            # print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            macds = visible_rows['macd']
            adlines = visible_rows['a/d']
            obvs = visible_rows['obv']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            if len(peak_indices) > 1 and len(valley_indices):
                corrected_indices, valley_indices, peak_indices = rearrange_valley_peak(valley_indices, valley_prices,
                                                                                        peak_indices, peak_prices,
                                                                                        prices.iloc[0])

                if valley_indices[-1] > peak_indices[-1]:  # from a valley
                    # if len(valley_indices) > 3:
                    #     _, low = standout(valley_prices[:-2])
                    #     high, _ = standout(valley_prices)
                    #     if low > len(valley_indices) * .618:
                    #         wavestart = valley_indices[-2]

                    wavestart = max(wavestart, valley_indices[-1] - 60)
                    last_sell = max([i for i, value in enumerate(positions) if value < 0], default=-1)
                    wavestart = max(wavestart, last_sell)
                    dips = valley_indices[valley_indices > wavestart]

                    for i in range(dips.size - 1):
                        patience = 0
                        start, end = dips[i], dips[-1]  # max wavelength, TODO: max magnitude
                        wavelength = end - start
                        if wavelength > 9:  # long enough to identify the trend
                            a_prices, _ = np.polyfit(np.arange(end - start), prices[start:end], 1)
                            a_macds, _ = np.polyfit(np.arange(end - start), macds[start:end], 1)
                            a_adlines, _ = np.polyfit(np.arange(end - start), adlines[start:end], 1)
                            a_obvs, _ = np.polyfit(np.arange(end - start), obvs[start:end], 1)
                            price_ratio = (prices.iloc[end] - prices.iloc[start]) / prices.iloc[start]
                            macd_ratio = (macds.iloc[end] - macds.iloc[start]) / abs(macds.iloc[start])
                            ad_ratio = (adlines.iloc[end] - adlines.iloc[start]) / abs(adlines.iloc[start])

                            if a_prices < 0 < a_macds and macd_ratio > 0 and (a_adlines > 0 or a_obvs > 0):
                                patience = end - start
                                break

                    if patience > 9:
                        if hold:
                            entry = valley_indices[-1]
                            print(f"entry changed to {entry}, patience changed to {patience} @ {count}")
                        if not hold and count < 386:
                            print(f"current dips: {dips}")
                            entry = valley_indices[-1]
                            position = 1
                            hold = True
                            buy_point = count
                            buy_price = price
                            print(f"buying @{count} {price} patience {patience} [{entry-patience}, {entry}] "
                              f"price roc: {price_ratio:.5f} "
                              f"ad roc: {ad_ratio:.5f} {a_adlines}")

            # from a price peak
            if hold:
                to_sell = False
                print(f"Peaks {peak_indices} Valleys {valley_indices} @ {count}")
                if peak_indices.size > 1 and valley_indices.size and peak_indices[-1] > valley_indices[-1]:
                    num_peaks = len(peak_indices[(peak_indices >= entry - patience) & (peak_indices <= entry)])
                    print(f"patience: {patience} @ {count} to be sold after {num_peaks} peaks since {entry}")

                    pokes = peak_indices[peak_indices > buy_point]
                    print(f"{pokes}")

                    # for i in range(pokes.size - 1):
                    #     start, end = pokes[i], pokes[-1]
                    #     price_ratio = (prices.iloc[end] - prices.iloc[start])
                    #     ad_ratio = (adlines.iloc[end] - adlines.iloc[start]) / abs(adlines.iloc[start])
                    #     if ad_ratio < 0 < price_ratio:
                    #         print(f"!!!!!!! {ad_ratio} < 0 < {price_ratio} @ {count}")
                    #         to_sell = True
                    #         break

                    if len(peak_indices[peak_indices > buy_point]) >= num_peaks:
                        print(f"{len(peak_indices)} {peak_indices[peak_indices > buy_point]} last peak {peak_indices[-1]}")
                        to_sell = True

                    if count == 389:
                        to_sell = True

                    if to_sell:
                        position = -1
                        hold = False
                        buy_price = 0
                        patience = 0
                        print(f"selling @{count} {row['close']} wavelength: {wavelength}")

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([20, 189], distance, prominence)

    def signal(self):
        self.deep_v()