from strategy import Strategy
from scipy.signal import find_peaks
import numpy as np
import pandas as pd


class RaftStrategy(Strategy):

    def raft(self):
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

        print(data[data['sma_volume_spike']].index)
        print(data[data['sma_volume_dip']].index)

        data['position'] = positions
        self.snapshot([290, -1], ['volume', 'gap'])

    def signal(self):
        self.raft()
