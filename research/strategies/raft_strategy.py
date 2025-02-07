from strategy import Strategy
from scipy.signal import find_peaks
import numpy as np


class RaftStrategy(Strategy):

    def raft(self):
        data = self.data
        positions = []
        data['position'] = 0
        count = 0
        buckets_in_use = 0
        distance = 3
        prominence = self.data.iloc[0]['close'] * 0.00125 + 0.005
        used_valley, used_peak = 0, 0
        checked_valley, checked_peak = 0, 0

        for index, row in data.iterrows():
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            highs, lows = visible_rows['high'], visible_rows['low']
            volumes = visible_rows['volume']

            if row['normalized_gap'] > 47:
                print(f"---- {count} {row['gap']} {row['normalized_gap']}!")
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            low_valleys, _ = find_peaks(-lows, distance=distance, prominence=prominence)
            volume_peaks, _ = find_peaks(volumes, distance=distance)
            volume_valleys, _ = find_peaks(-volumes, distance=distance)

            valleys = np.union1d(valleys, low_valleys)

            if len(peaks) > 1 and len(valleys) > 1 and len(volume_peaks) and len(volume_valleys):

                valley, prior_valley, peak, prior_peak = valleys[-1], valleys[-2], peaks[-1], peaks[-2]
                spike_mean = int(volumes.iloc[volume_peaks].mean()) // 1000
                drop_mean = int(volumes.iloc[volume_valleys].mean()) // 1000

                valleys = valleys[valleys > peak]
                peaks = peaks[peaks > valley]

                for valley in valleys:
                    valley_volume = volumes.iloc[valley] // 1000
                    if valley == 30:
                        print(f"----- {count} [{drop_mean} - {spike_mean}] {valley_volume} valley@{valley}")
                        print(f"----- {visible_rows.iloc[30]['volume'] // 1000}")

                    if valley > peak and buckets_in_use < self.num_buckets and valley != used_valley:  # from a valley
                        if valley_volume > spike_mean and valley > checked_valley:
                            print(f"{count} [{drop_mean} - {spike_mean}] {valley_volume} valley@{valley}")
                            checked_valley = valley

                        if valley == checked_valley:
                            if True or 'CDL_BELTHOLD' not in visible_rows.iloc[valley]['candlestick'] or visible_rows.iloc[valley]['open'] < visible_rows.iloc[valley]['close']:
                                used_valley = valley
                                print(f"{visible_rows.iloc[valley]['candlestick']} @ {valley}")
                                position = 1
                                buckets_in_use += 1
                                if buckets_in_use > self.num_buckets:
                                    buckets_in_use = self.num_buckets
                                print(f"---- buy signal @ {count} from valley {valley}")

                for peak in peaks:
                    peak_volume = volumes.iloc[peak] // 1000
                    if valley < peak != used_peak and buckets_in_use:
                        if peak_volume < drop_mean and peak > checked_peak:
                            checked_peak = peak
                            print(f"{count} [{drop_mean} - {spike_mean}] {peak_volume} peak@{peak}")
                            print(f"---- sell signal @ {count} after peak {peak}")
                            position = -1
                            buckets_in_use -= 1
                            if buckets_in_use < 0:
                                buckets_in_use = 0
                            used_peak = peak

                stop_loss = price > prices.iloc[used_valley] * 1.02 or price < prices.iloc[used_valley] * 0.09
                if buckets_in_use and stop_loss:
                    position = -4
                    buckets_in_use -= 4
                    if buckets_in_use < 0:
                        buckets_in_use = 0
                    used_peak = peak

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([60, 120], ['gap', 'normalized_gap'])

    def signal(self):
        self.raft()
