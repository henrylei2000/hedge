from strategy import Strategy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
            data['a/d'] = Strategy.ad_line(data['close'], data['high'], data['low'], data['volume'])
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
                corrected_indices, valley_indices, peak_indices = Strategy.rearrange_valley_peak(valley_indices, valley_prices,
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
                obv_corrected_indices, obv_valley_indices, obv_peak_indices = Strategy.rearrange_valley_peak(obv_valley_indices,
                                                                                                    obv_valley_prices,
                                                                                                    obv_peak_indices,
                                                                                                    obv_peak_prices,
                                                                                                    obvs.iloc[0])
                obv_peak_prices = obvs.iloc[obv_peak_indices]
                obv_valley_prices = obvs.iloc[obv_valley_indices]

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
        self.snapshot([0, 159])

    def signal(self):
        self.trend()
