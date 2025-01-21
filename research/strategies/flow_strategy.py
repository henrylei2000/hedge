from strategy import Strategy
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def ad_line(prices, high, low, volume):
    # Money Flow Multiplier calculation
    money_flow_multiplier = ((prices - low) - (high - prices)) / (high - low)
    # Money Flow Volume calculation
    money_flow_volume = money_flow_multiplier * volume
    # Accumulation/Distribution Line (cumulative sum of money flow volume)
    ad_line = money_flow_volume.cumsum()
    return ad_line


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

    def flow(self):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        count = 0
        buckets_in_use = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

        for index, row in data.iterrows():
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            volumes = visible_rows['volume']
            macds = visible_rows['macd']
            rsis = visible_rows['rsi']
            adlines = visible_rows['a/d']
            obvs = visible_rows['obv']

            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            volume_peaks, _ = find_peaks(volumes)

            if len(peaks) > 1 and len(valleys) > 1:
                valley, prior_valley, peak, prior_peak = valleys[-1], valleys[-2], peaks[-1], peaks[-2]
                if valley > peak and buckets_in_use < self.num_buckets:  # from a valley

                    reversal = 0
                    dips = valleys[valleys > valley - 60]
                    is_macd_divergence = False
                    for i in range(dips.size - 1):
                        start, end = dips[i], dips[-1]
                        a_prices, _ = np.polyfit(np.arange(end - start), prices[start:end], 1)
                        a_macds, _ = np.polyfit(np.arange(end - start), macds[start:end], 1)
                        a_adlines, _ = np.polyfit(np.arange(end - start), adlines[start:end], 1)
                        a_obvs, _ = np.polyfit(np.arange(end - start), obvs[start:end], 1)

                        if a_prices < 0 < a_macds:
                            is_macd_divergence = True
                            print(f"{valley} - {count} macd divergence found!")
                            break

                    is_volume_pike = False
                    pre_volume = volumes.iloc[prior_valley:valley - 1].mean()
                    post_volume = volumes.iloc[valley - 1:valley + 1].max()
                    if post_volume > pre_volume * 1.5:  # at least 1.5x average volume
                        is_volume_pike = True

                    if is_macd_divergence and is_volume_pike:
                        reversal += 1
                        if rsis.iloc[valley] < 30:
                            reversal += 1

                    if reversal > 0:
                        position = reversal
                        buckets_in_use += reversal
                        if buckets_in_use > self.num_buckets:
                            buckets_in_use = self.num_buckets
                        print(f"---- buy signal scored {reversal} @ {count}")

                if peak > valley and buckets_in_use:
                    to_sell = 0
                    if rsis.iloc[peak] > 75 and rsi < rsis.iloc[peak]:
                        to_sell += 1
                    elif rsi > rsis.iloc[peak]:
                        to_sell -= 1

                    pre_macd = macds.iloc[max(0, peak - 5):peak].mean()
                    post_macd = macds.iloc[peak + 1:count].mean()
                    if strength < 0 and post_macd < pre_macd:
                        to_sell += 1
                    else:
                        to_sell -= 1
                    if to_sell > 0:
                        print(f"---- sell signal scored {to_sell} @ {count}")
                        position = -to_sell
                        buckets_in_use -= to_sell
                        if buckets_in_use < 0:
                            buckets_in_use = 0

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([80, 200], distance, prominence)

    def deep_v(self, lookback=5, lookforward=3, decline_threshold=0.003, recovery_threshold=0.0035):
        self.flow_simple()
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        count = 0

        buckets_in_use = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

        for index, row in data.iterrows():
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
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

                valley, peak = valley_indices[-1], peak_indices[-1]
                if valley > peak and len(prices) < valley + lookforward and buckets_in_use < self.num_buckets:  # from a valley
                    deep_v = 0

                    pre_volume = volumes.iloc[max(0, valley - lookback):valley - 1].mean()
                    post_volume = volumes.iloc[valley - 1:valley + lookforward].max()
                    if post_volume > pre_volume * 1.5:  # at least 1.5x average volume
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
                        buckets_in_use += deep_v
                        if buckets_in_use > self.num_buckets:
                            buckets_in_use = self.num_buckets
                        print(f"---- buy signal scored {deep_v} @ {count}")

                if peak > valley and buckets_in_use:
                    to_sell = 0
                    if rsis.iloc[peak] > 75 and rsi < rsis.iloc[peak]:
                        to_sell += 1
                    elif rsi > rsis.iloc[peak]:
                        to_sell -= 1

                    pre_macd = macds.iloc[max(0, peak - lookback):peak].mean()
                    post_macd = macds.iloc[peak + 1:count].mean()
                    if strength < 0 and post_macd < pre_macd:
                        to_sell += 1
                    else:
                        to_sell -= 1
                    if to_sell > 0:
                        print(f"---- sell signal scored {to_sell} @ {count}")
                        position = -to_sell
                        buckets_in_use -= to_sell
                        if buckets_in_use < 0:
                            buckets_in_use = 0

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([80, 200], distance, prominence)

    def snapshot(self, interval, distance, prominence):
        if interval[1] - interval[0] < 30:
            return
        rows = self.data.iloc[interval[0]:interval[1]]
        prices = rows['close']

        # Identify peaks and valleys
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        peak_indices = np.array(peaks)
        valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
        valley_indices = np.array(valleys)

        indicator = 'macd'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.00125 + 0.005
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance*3, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_valleys, _ = find_peaks(-obvs, distance=distance*3, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)

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

        indicator = 'volume'
        obvs = rows[indicator]
        obv_prominence = self.data.iloc[0][indicator] * 0.00125 + 0.005
        # Identify peaks and valleys
        obv_peaks, _ = find_peaks(obvs, distance=distance, prominence=obv_prominence)
        obv_peak_indices = np.array(obv_peaks)
        obv_valleys, _ = find_peaks(-obvs, distance=distance, prominence=obv_prominence)
        obv_valley_indices = np.array(obv_valleys)
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
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)

            if len(peaks) > 1 and len(valleys):
                if valleys[-1] > peaks[-1]:  # from a valley
                    wavestart = max(wavestart, valleys[-1] - 60)
                    last_sell = max([i for i, value in enumerate(positions) if value < 0], default=-1)
                    wavestart = max(wavestart, last_sell)
                    dips = valleys[valleys > wavestart]

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
                            entry = valleys[-1]
                            print(f"entry changed to {entry}, patience changed to {patience} @ {count}")
                        if not hold and count < 386:
                            print(f"current dips: {dips}")
                            entry = valleys[-1]
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
                print(f"Peaks {peaks} Valleys {valleys} @ {count}")
                if len(peaks) > 1 and len(valleys) and peaks[-1] > valleys[-1]:
                    num_peaks = len(peaks[(peaks >= entry - patience) & (peaks <= entry)])
                    print(f"patience: {patience} @ {count} to be sold after {num_peaks} peaks since {entry}")

                    pokes = peaks[peaks > buy_point]
                    print(f"{pokes}")

                    if len(peaks[peaks > buy_point]) >= num_peaks:
                        print(f"{len(peaks)} {peaks[peaks > buy_point]} last peak {peaks[-1]}")
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
        self.flow()