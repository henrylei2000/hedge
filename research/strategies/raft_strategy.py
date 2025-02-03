from strategy import Strategy
from scipy.signal import find_peaks


class RaftStrategy(Strategy):

    def raft(self):
        data = self.data
        positions = []
        data['position'] = 0
        count = 0
        buckets_in_use = 0
        distance = 3
        used_valley, used_peak = 0, 0
        checked_valley, checked_peak = 0, 0

        for index, row in data.iterrows():
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            volumes = visible_rows['volume']

            peaks, _ = find_peaks(prices, distance=distance)
            valleys, _ = find_peaks(-prices, distance=distance)
            volume_peaks, _ = find_peaks(volumes, distance=distance)
            volume_valleys, _ = find_peaks(-volumes, distance=distance)

            if len(peaks) > 1 and len(valleys) > 1:

                valley, prior_valley, peak, prior_peak = valleys[-1], valleys[-2], peaks[-1], peaks[-2]

                if len(visible_rows[visible_rows['volume_spike']]):
                    spike_mean = int(visible_rows[visible_rows['volume_spike']]['volume'].mean()) // 1000
                else:
                    spike_mean = int(visible_rows['volume'].mean()) // 1000
                if len(visible_rows[visible_rows['volume_drop']]):
                    drop_mean = int(visible_rows[visible_rows['volume_drop']]['volume'].mean()) // 1000
                else:
                    drop_mean = int(visible_rows['volume'].mean()) // 1000
                valley_volume = volumes.iloc[valley] // 1000
                peak_volume = volumes.iloc[peak] // 1000

                if valley > peak and buckets_in_use < self.num_buckets and valley != used_valley:  # from a valley
                    if (valley_volume > spike_mean or valley_volume < drop_mean) and valley > checked_valley:
                        print(f"{count} spike {spike_mean}, drop {drop_mean}, {valley_volume} @ {valley}")
                        checked_valley = valley

                    if valley == checked_valley:
                        position = 1
                        buckets_in_use += 1
                        if buckets_in_use > self.num_buckets:
                            buckets_in_use = self.num_buckets
                        used_valley = valley
                        print(f"---- buy signal @ {count} from valley {valley}")

                if valley < peak != used_peak and buckets_in_use:
                    if (peak_volume > spike_mean or peak_volume < drop_mean) and peak > checked_peak:
                        checked_peak = peak
                        print(f"{count} spike {spike_mean}, drop {drop_mean}, {peak_volume} @ {peak}")
                        print(f"---- sell signal @ {count}")
                        position = -1
                        buckets_in_use -= 1
                        if buckets_in_use < 0:
                            buckets_in_use = 0
                        used_peak = peak

                stop_loss = price > prices.iloc[used_valley] * 1.015 or price < prices.iloc[used_valley] * 0.085
                if buckets_in_use and stop_loss:
                    position = -4
                    buckets_in_use -= 4
                    if buckets_in_use < 0:
                        buckets_in_use = 0
                    used_peak = peak

            positions.append(position)
            count += 1

        data['position'] = positions
        self.snapshot([260, 330], ['volume', 'gap'])

    def signal(self):
        self.raft()
