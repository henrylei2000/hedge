from strategy import Strategy
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
        high = price_data[most_recent_peak]
        low = price_data[most_recent_valley]
        recent_fib_levels = {f"{int(ratio * 100)}%": high - (high - low) * ratio for ratio in
                             [0.236, 0.382, 0.5, 0.618, 1.0]}

    elif most_recent_peak is not None and most_recent_valley is not None and most_recent_peak > most_recent_valley:
        high = price_data[most_recent_peak]
        low = price_data[most_recent_valley]
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


class FlowStrategy(Strategy):
    def flow_simple(self):
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
            prices = visible_rows['close']
            macds = visible_rows['macd']
            ads = visible_rows['a/d']

            macd_divergence = calculate_recent_macd_divergence_with_crossover(prices, macds, divergence_window)
            ad_divergence = detect_recent_ad_line_divergence(prices, ads, divergence_window)

            peaks, _ = find_peaks(prices, distance=divergence_window)
            valleys, _ = find_peaks(-prices, distance=divergence_window)
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

    def snapshot(self, interval, indicator, distance, prominence):
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

        # indicator = 'a/d'
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
        self.flow_simple()
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
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

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
                peak_prices = prices.iloc[peak_indices]
                valley_prices = prices.iloc[valley_indices]

            indicator = 'rsi'
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

            # from a valley
            if valley_indices.size > 1 and peak_indices.size and valley_indices[-1] > peak_indices[-1]:
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

                            if price_ratio < 0 < ad_ratio and price_ratio * ad_ratio < -0.01:  # TODO: alternative way to describe the divergence
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
                print(f"Peaks {peak_indices} Valleys {valley_indices} @ {count}")
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
        self.snapshot([0, 159], indicator, distance, prominence)

    def signal(self):
        self.trend()
