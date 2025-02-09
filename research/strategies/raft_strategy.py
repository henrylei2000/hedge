from strategy import Strategy
from scipy.signal import find_peaks
import numpy as np


class RaftStrategy(Strategy):

    def raft(self):
        data = self.data
        self.normalized('gap')
        self.normalized('volume')
        self.normalized('span')
        self.normalized('variance')
        positions = []
        data['position'] = 0
        buckets_in_use = 0
        distance = 3
        prominence = self.data.iloc[0]['close'] * 0.00125 + 0.005
        used_valley, used_peak = 0, 0
        checked_valley, checked_peak = 0, 0

        signals = []

        # self.snapshot([0, 120], ['normalized_span', 'normalized_variance'])

        for index in data.index:
            row = data.iloc[index]
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            highs, lows = visible_rows['high'], visible_rows['low']
            volumes = visible_rows['volume']

            print(f"{index:3d} trending {row['normalized_variance']:4d} with volume {row['normalized_volume']:3d}", end=" ")

            if index > 5 and row['normalized_variance'] < -70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                print("***")
                signals.append(index)
                if row['normalized_variance'] > -90:
                    self.normalized('variance', index)
                if row['normalized_volume'] > 90:
                    self.normalized('volume', index)
            else:
                print()

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

                    if valley > peak and buckets_in_use < self.num_buckets and valley != used_valley:  # from a valley
                        if valley_volume > spike_mean and valley > checked_valley:
                            # print(f"{index} [{drop_mean} - {spike_mean}] {valley_volume} valley@{valley}")
                            # print(f"---- buy signal @ {index} from valley {valley}")
                            checked_valley = valley

                        if valley == checked_valley:
                            used_valley = valley
                            position = 1
                            buckets_in_use += 1
                            if buckets_in_use > self.num_buckets:
                                buckets_in_use = self.num_buckets

                for peak in peaks:
                    peak_volume = volumes.iloc[peak] // 1000
                    if valley < peak != used_peak and buckets_in_use:
                        if peak_volume < drop_mean and peak > checked_peak:
                            checked_peak = peak
                            # print(f"{index} [{drop_mean} - {spike_mean}] {peak_volume} peak@{peak}")
                            # print(f"---- sell signal @ {index} after peak {peak}")
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

        positions = [0] * len(data)
        for i in signals:
            positions[i] = 1
        data['position'] = positions

        print(signals)

    def signal(self):
        self.raft()
