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


def smooth(prices, valleys, peaks):

    # Thresholds
    min_price_change = 0.5  # Minimum price change to consider significant
    min_time_diff = 3  # Minimum time (or index) difference between peaks/valleys

    # Initialize lists to store the significant peaks and valleys
    significant_valleys = []
    significant_peaks = []
    filtered_valleys = np.array([])
    filtered_peaks = np.array([])

    # Step 1: Price change filtering
    for i in range(min(len(peaks), len(valleys))):
        idx1, idx2 = valleys[i], peaks[i]
        # Calculate price change between the peak and the valley
        price_change = abs(prices.iloc[idx2] - prices.iloc[idx1])

        # Only keep the peak-valley pair if the price change is significant
        if price_change >= min_price_change:
            significant_valleys.append(idx1)
            significant_peaks.append(idx2)

    if len(significant_valleys) or len(significant_peaks):
        # Convert lists to arrays
        significant_valleys = np.array(significant_valleys)
        significant_peaks = np.array(significant_peaks)

        # Step 2: Minimum time filtering (for both peaks and valleys)
        filtered_valleys = [significant_valleys[0]]  # Start with the first valley
        filtered_peaks = [significant_peaks[0]]  # Start with the first peak

        for i in range(1, len(significant_valleys)):
            if significant_valleys[i] - filtered_valleys[-1] >= min_time_diff:
                filtered_valleys.append(significant_valleys[i])

        for i in range(1, len(significant_peaks)):
            if significant_peaks[i] - filtered_peaks[-1] >= min_time_diff:
                filtered_peaks.append(significant_peaks[i])

        # Convert to arrays
        filtered_valleys = np.array(filtered_valleys)
        filtered_peaks = np.array(filtered_peaks)

    return filtered_valleys, filtered_peaks


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


def ad_line(prices, high, low, volume):
    # Money Flow Multiplier calculation
    money_flow_multiplier = ((prices - low) - (high - prices)) / (high - low)

    # Money Flow Volume calculation
    money_flow_volume = money_flow_multiplier * volume

    # Accumulation/Distribution Line (cumulative sum of money flow volume)
    ad_line = money_flow_volume.cumsum()
    return ad_line


class WaveStrategy(Strategy):
    def wave_simple(self):
        short_window, long_window, signal_window = 9, 21, 6  # 12, 26, 9
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
            data['rolling_volume'] = data['obv'].rolling(window=3).mean()
            data['a/d'] = ad_line(data['close'], data['high'], data['low'], data['volume'])
            self.normalize_column('a/d')
            # Generate Buy and Sell signals
            data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
            # data.to_csv(f"{self.symbol}.csv")

    def normalize_column(self, column_name):
        # Extract the column to be normalized
        col = self.data[column_name]

        # Normalize the column to range from 0 to 100
        normalized_col = (col - col.min()) / (col.max() - col.min()) * 100

        # Create a new column with the normalized values
        self.data['normalized_' + column_name] = normalized_col

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
        # valley_indices, peak_indices = smooth(prices, valley_indices, peak_indices)
        peak_prices = prices.iloc[peak_indices]
        valley_prices = prices.iloc[valley_indices]

        indicator = 'a/d'
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

        if len(obv_peak_indices):
            # Perform linear regression on peaks
            obv_a_peaks, obv_b_peaks = np.polyfit(obv_peak_indices[-5:], obv_peak_prices[-5:], 1)
            # Perform linear regression on valleys
            obv_a_valleys, obv_b_valleys = np.polyfit(obv_valley_indices[-5:], obv_valley_prices[-5:], 1)

        # debugging info
        dips = []
        for i in range(valley_indices.size - 1):
            high, low = standout(valley_prices.iloc[:i+1])
            # print(f"{valley_indices[i]} {valley_prices.iloc[i]} ({high}, {low})")
            if valley_indices[i] > valley_indices[-1] - 60:
                dips.append((valley_indices[i], valley_indices[-1]))

        print(f"[SNAPSHOT] {dips}")

        # Get positions for buy (1) and sell (-1) signals
        buy_signals = rows[rows['position'] == 1]
        sell_signals = rows[rows['position'] == -1]
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

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

        ax2.plot(rows[indicator], label=f"{indicator}", color='purple')
        ax2.plot(obvs.iloc[obv_peak_indices], 'ro', label='Peaks')
        # Annotate each peak with its value
        for peak in obv_peak_indices:
            ax2.annotate(f'{interval[0] + peak}',
                         (obvs.index[peak], obvs.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9)  # You can adjust the font size if needed
        ax2.plot(obvs.iloc[obv_valley_indices], 'go', label='Valleys')
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
        ax2.set_title(f"{indicator}")
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f"{indicator}")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def trend(self):
        self.wave_simple()
        data = self.data
        positions = []
        data['position'] = 0

        obv_bullish, macd_bullish, price_bullish = False, False, False
        hold = False
        wavelength, wavestart, entry, patience, next_peak = 0, 0, 0, 0, False
        buy_price = 0
        a_peaks = 1000000
        b_peaks = 1000000
        count = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.0039 + 0.0047

        for index, row in data.iterrows():
            position = 0
            price = row['close']
            # print(f"[{index.strftime('%Y-%m-%d %H:%M:%S')} {price:.4f} @ {count}]")
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']
            adlines = visible_rows['a/d']

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
                # valley_indices, peak_indices = smooth(prices, valley_indices, peak_indices)
                peak_prices = prices.iloc[peak_indices]
                valley_prices = prices.iloc[valley_indices]

            indicator = 'a/d'
            obvs = visible_rows[indicator]
            obv_prominence = self.data.iloc[0][indicator] * 0.1

            previous_avg = np.mean(obvs[:index])
            if row['volume'] > 7 * previous_avg:
                pass
                # print(f"Spike found {row['volume']} vs {previous_avg}")

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

            if len(peak_indices) > 1:
                # Perform linear regression on peaks
                a_peaks, b_peaks = np.polyfit(peak_indices[-5:], peak_prices[-5:], 1)
                peak_projected = a_peaks * count + b_peaks

            if len(valley_indices) > 1:
                # Perform linear regression on valleys
                a_valleys, b_valleys = np.polyfit(valley_indices[-5:], valley_prices[-5:], 1)
                valley_projected = a_valleys * count + b_valleys

            # from a valley
            if valley_indices.size > 1 and peak_indices.size and valley_indices[-1] > peak_indices[-1]:
                high, low = standout(valley_prices)
                if low > 5:
                    wavestart = valley_indices[-1]
                wavestart = max(wavestart, valley_indices[-1] - 60)
                dips = valley_indices[valley_indices > wavestart]  # TODO: fake wave!
                if len(dips) < 5:
                    dips = valley_indices[-3:]
                best_ad, selected_pos = 0, 0
                for i in range(dips.size - 1):
                        start, end = dips[i], dips[-1]
                        wavelength = end - start
                        if wavelength > 3:
                            a_prices, _ = np.polyfit(np.arange(end - start), prices[start:end], 1)
                            a_adlines, _ = np.polyfit(np.arange(end - start), adlines[start:end], 1)
                            price_ratio = (prices.iloc[end] - prices.iloc[start])
                            ad_ratio = (adlines.iloc[end] - adlines.iloc[start]) / abs(adlines.iloc[start])

                            if price_ratio < 0 < ad_ratio and price_ratio * ad_ratio < - 0.01:  # TODO: alternative way to describe the divergence
                                interval = wavelength // 3
                                if adlines[start:start + interval].sum() < adlines[start + interval*2:end].sum():
                                    if ad_ratio > best_ad:
                                        best_ad = ad_ratio
                                        patience = end - start
                                    if 20 < count < 30:
                                        print(f"{a_prices} < 0 < {a_adlines} with a divergence of {price_ratio * ad_ratio:.5f}")
                if 20 < count < 30:
                    print(f"best ad ratio {best_ad} @ {count}")
                if best_ad > 0:
                    if hold and price < buy_price:
                        entry = valley_indices[-1]
                        print(f"entry changed to {entry}, patience changed to {patience} @ {count}")
                    if not hold:
                        entry = valley_indices[-1]
                        position = 1
                        hold = True
                        buy_price = price
                        print(f"buying @{count} {price} patience {patience} [{entry-patience}, {entry}] "
                          f"price roc: {price_ratio:.5f} "
                          f"ad roc: {ad_ratio:.5f} {a_adlines}")

            # from a price peak
            if hold:
                to_sell = False
                if peak_indices.size > 1 and valley_indices.size and peak_indices[-1] > valley_indices[-1]:
                    num_peaks = len(peak_indices[(peak_indices > entry - patience) & (peak_indices < entry)])
                    print(
                        f"patience: {patience} @ {count} to be sold after {num_peaks} peaks since {entry}")
                    if 20 < count < 30:
                        print(f"{peak_indices[-5:]} {num_peaks} {entry} {patience} @ {count}")

                    pokes = peak_indices[peak_indices > entry]
                    print(f"{pokes}")
                    for i in range(pokes.size - 1):
                        start, end = pokes[i], pokes[-1]
                        price_ratio = (prices.iloc[end] - prices.iloc[start])
                        ad_ratio = (adlines.iloc[end] - adlines.iloc[start]) / abs(adlines.iloc[start])
                        if ad_ratio < 0 < price_ratio:
                            print(f"!!!!!!! {ad_ratio} < 0 < {price_ratio} @ {count}")
                            to_sell = True
                            break

                    if len(peak_indices[peak_indices > entry]) >= num_peaks:
                        print(f"{len(peak_indices)} {peak_indices[peak_indices > entry]} last peak {peak_indices[-1]}")
                        to_sell = True

                    if to_sell:
                        position = -1
                        hold = False
                        buy_price = 0
                        print(f"selling @{count} {row['close']} wavelength: {wavelength}")

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([0, 119], distance, prominence)

    def signal(self):
        self.trend()
