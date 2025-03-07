from strategy import Strategy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class WaveStrategy(Strategy):

    def flow(self):
        data = self.data
        positions = []
        data['position'] = 0
        count = 0
        buckets_in_use = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005
        used_valley, used_peak = 0, 0
        volume_spikes = pd.DataFrame(columns=['sma_volume_spike'])
        volume_dips = pd.DataFrame(columns=['sma_volume_dip'])

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
                if valley > peak and buckets_in_use < self.num_buckets and valley != used_valley:  # from a valley

                    # Wave 1 potential zone: Check if price is moving higher from the valley
                    valley_price = prices.iloc[valley]
                    if price > valley_price:  # Upward movement detected
                        # Confirm if price has exceeded the previous peak
                        if len(peaks) >= 2:
                            if price > prices.iloc[prior_peak]:
                                # MACD Divergence Confirmation
                                macd_valley = macds.iloc[valley]
                                macd_prior_valley = macds.iloc[prior_valley]

                                # Check for bullish divergence
                                if macd_valley > macd_prior_valley:
                                    # Signal a potential entry for Wave 1
                                    pass

                    start, end, reversal = 0, 0, 0
                    dips = valleys[valleys > valley - 30]
                    is_macd_divergence = False
                    for i in range(dips.size - 1):
                        start, end = dips[i], dips[-1]
                        a_prices, _ = np.polyfit(np.arange(end - start), prices[start:end], 1)
                        a_macds, _ = np.polyfit(np.arange(end - start), macds[start:end], 1)
                        a_adlines, _ = np.polyfit(np.arange(end - start), adlines[start:end], 1)
                        a_obvs, _ = np.polyfit(np.arange(end - start), obvs[start:end], 1)

                        if a_prices < 0 < a_macds:
                            is_macd_divergence = True
                            print(f"{valley} - {count} macd divergence found: {start} - {end}!")
                            break

                    if is_macd_divergence:
                        reversal += 1
                        if rsis.iloc[valley] < 30:
                            reversal += 1
                            print(f"rsi + 1")

                    if reversal > 0:
                        position = reversal
                        buckets_in_use += reversal
                        if buckets_in_use > self.num_buckets:
                            buckets_in_use = self.num_buckets
                        used_valley = valley
                        print(f"---- buy signal scored {reversal} @ {count} {end} {valley}")

                if valley < peak != used_peak and buckets_in_use:
                    to_sell = 0
                    is_volume_decline = False
                    pre_volume = volumes.iloc[used_valley:peak - 1].mean()
                    post_volume = volumes.iloc[peak]
                    if post_volume < pre_volume * 0.75:
                        is_volume_decline = True
                        print(f"volume decline {count}")

                    rolling_max = macds.iloc[count - 9:count].max() if count >= 9 else macds.iloc[:count].max()
                    is_macd_bearish = macds.iloc[count] < rolling_max
                    if is_macd_bearish:
                        print(f"macd bearish @{count} after {peak}")

                    if rsis.iloc[peak] > 75:
                        to_sell += 1
                    if adlines.iloc[peak] < adlines.iloc[prior_peak]:
                        to_sell += 1

                    if is_volume_decline and is_macd_bearish and to_sell > 0:
                        print(f"---- sell signal scored {to_sell} @ {count}")
                        position = -to_sell
                        buckets_in_use -= to_sell
                        if buckets_in_use < 0:
                            buckets_in_use = 0
                        used_peak = peak

                if buckets_in_use and (
                        price > prices.iloc[used_valley] * 1.015 or price < prices.iloc[used_valley] * 0.085):
                    print(f"---- sell signal scored FULL @ {count}")
                    position = -4
                    buckets_in_use -= 4
                    if buckets_in_use < 0:
                        buckets_in_use = 0
                    used_peak = peak

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([290, -1])

    def wave3(self):
        data = self.data
        positions = []
        data['position'] = 0
        count = 0
        buckets_in_use = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005
        used_valley, used_peak, wave_1_length = 0, 0, 0

        for index, row in data.iterrows():
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            volumes = visible_rows['volume']
            macds = visible_rows['macd']
            signals = visible_rows['signal_line']
            rsis = visible_rows['rsi']
            adlines = visible_rows['a/d']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            corrected_indices, valley_indices, peak_indices = Strategy.rearrange_valley_peak(valley_indices,
                                                                                             valley_prices,
                                                                                             peak_indices, peak_prices,
                                                                                             prices.iloc[0])

            if len(peak_indices) > 1 and len(valley_indices) > 1:
                # Use the most recent two peaks and valleys
                valley, prior_valley = valley_indices[-1], valley_indices[-2]
                peak, prior_peak = peak_indices[-1], peak_indices[-2]
                # Validate Wave 1 and Wave 2 structure
                if valley > peak and prices.iloc[valley] > prices.iloc[prior_valley]:
                    wave_1_start = prior_valley  # Assume the prior valley is the start of Wave 1
                    wave_1_length = prices.iloc[peak] - prices.iloc[wave_1_start]
                    wave_2_retracement = (prices.iloc[peak] - prices.iloc[
                        valley]) / wave_1_length if wave_1_length else 0

                    if 0.2 <= wave_2_retracement <= 0.8:
                        # Validate momentum and volume for Wave 3 entry
                        is_macd_bullish = macds.iloc[count] > macds.iloc[valley] > 0 or macds.iloc[count] > \
                                          signals.iloc[count]  # MACD bullish crossover

                        volume_wave_1 = volumes[wave_1_start:peak + 1].mean()
                        volume_wave_2 = volumes[peak:valley].mean()
                        volume_wave_3 = volumes[valley:].mean()
                        is_volume_increasing = volume_wave_3 > volume_wave_1 and volume_wave_3 > volume_wave_2

                        if is_macd_bullish and is_volume_increasing and valley > used_valley:
                            if price <= prices.iloc[peak] or macds.iloc[count] <= macds.iloc[peak] * 0.995:
                                continue
                            # Confirm breakout above Wave 1 peak
                            print(f"{price} > {prices.iloc[peak]} {peak}")
                            # Add to Wave 3 entries
                            position = 4
                            buckets_in_use += 4
                            if buckets_in_use > self.num_buckets:
                                buckets_in_use = self.num_buckets
                            used_valley = valley
                            print(f"Wave 3 start detected at index {count}, price: {prices.iloc[count]}")

                if valley < peak != used_peak and buckets_in_use:
                    to_sell = 0

                    rolling_max = macds.iloc[count - 9:count].max() if count >= 9 else macds.iloc[:count].max()
                    is_macd_bearish = macds.iloc[count] < rolling_max
                    if is_macd_bearish:
                        to_sell += 4

                    if rsis.iloc[peak] > 75:
                        to_sell += 4
                    if adlines.iloc[peak] < adlines.iloc[prior_peak]:
                        to_sell += 4

                    if is_macd_bearish and to_sell > 0 and price > prices.iloc[used_valley] + wave_1_length * 1.618:
                        print(f"---- sell signal scored {to_sell} @ {count}")
                        position = -to_sell
                        buckets_in_use -= to_sell
                        if buckets_in_use < 0:
                            buckets_in_use = 0
                        used_peak = peak

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([200, 300], ['rsi', 'volume'])

    def divergence(self):
        data = self.data
        positions = []
        data['position'] = 0
        hold = False
        wavelength, wavestart, entry, patience, next_peak = 0, 0, 0, 0, False
        buy_point = 0
        count = 0
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005

        for index, row in data.iterrows():
            position = 0
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
                        start, end = dips[i], dips[-1]  # max wavelength, max magnitude
                        wavelength = end - start
                        if wavelength > 9:  # long enough to identify the trend
                            a_prices, _ = np.polyfit(np.arange(end - start), prices[start:end], 1)
                            a_macds, _ = np.polyfit(np.arange(end - start), macds[start:end], 1)
                            a_adlines, _ = np.polyfit(np.arange(end - start), adlines[start:end], 1)
                            a_obvs, _ = np.polyfit(np.arange(end - start), obvs[start:end], 1)
                            macd_ratio = (macds.iloc[end] - macds.iloc[start]) / abs(macds.iloc[start])

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
                        patience = 0
                        print(f"selling @{count} {row['close']} wavelength: {wavelength}")

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([0, 100])

    def trend(self):
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

                            if price_ratio < 0 < ad_ratio and price_ratio * ad_ratio < -0.01:
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
        self.flow()
