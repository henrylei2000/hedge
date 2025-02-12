from strategy import Strategy


class RaftStrategy(Strategy):

    def analyze_candlesticks(self):
        data = self.data
        signals = []

        for idx, row in data.iterrows():
            upper = int(row['upper_wick'] * 100)
            body = int(row['body_size'] * 100)
            lower = int(row['lower_wick'] * 100)
            span = row['normalized_span']
            volume = row['normalized_volume']
            signal = []

            # Strong move (large body with high volume confirms significance)
            if abs(body) > 70 and span > 50:
                direction = "bullish" if body > 0 else "bearish"
                if volume > 60:
                    signal.append(f"Strong {direction} move with high volume")
                else:
                    signal.append(f"Strong {direction} move but low volume")

            # Resistance (long upper wick with high volume strengthens the signal)
            if upper > 30 and span > 50:
                if volume > 60:
                    signal.append("Potential strong resistance (long upper wick, high volume)")
                else:
                    signal.append("Potential weak resistance (long upper wick, low volume)")

            # Support (long lower wick with high volume confirms demand)
            if lower > 30 and span > 50:
                if volume > 60:
                    signal.append("Potential strong support (long lower wick, high volume)")
                else:
                    signal.append("Potential weak support (long lower wick, low volume)")

            # Reversal signs with volume consideration
            if idx > 0:
                prev_row = data.iloc[idx - 1]
                prev_upper = int(prev_row['upper_wick'] * 100)
                prev_body = int(prev_row['body_size'] * 100)
                prev_lower = int(prev_row['lower_wick'] * 100)
                prev_span = prev_row['normalized_span']
                prev_volume = prev_row['normalized_volume']

                # Bullish reversal: long lower wick after a down candle with strong volume
                if prev_body < 0 < body and lower > 20 and span > 50 and volume > 50:
                    signal.append("Bullish reversal signal confirmed by high volume")

                # Bearish reversal: long upper wick after an up candle with strong volume
                if prev_body > 0 > body and upper > 20 and span > 50 and volume > 50:
                    signal.append("Bearish reversal signal confirmed by high volume")

            # Consolidation pattern (small bodies with low volume suggests indecision)
            if idx > 2:
                last_three_bodies = [abs(data.iloc[j]['body_size']) * 100 for j in range(idx - 2, idx + 1)]
                last_three_spans = [data.iloc[j]['normalized_span'] for j in range(idx - 2, idx + 1)]
                last_three_volumes = [data.iloc[j]['normalized_volume'] for j in range(idx - 2, idx + 1)]
                if all(b < 20 for b in last_three_bodies) and all(s < 50 for s in last_three_spans) and all(
                        v < 40 for v in last_three_volumes):
                    signal.append("Possible low-volume consolidation (low volatility, low volume)")

            signals.append((idx, ", ".join(signal) if signal else ""))

        self.data['candlestick'] = data.index.map(dict(signals))
        return signals

    def raft(self):
        data = self.data
        data['position'] = 0
        buckets_in_use = 0
        entries, exits = [], []
        self.normalized('variance')
        self.normalized('volume')
        self.normalized('tension')
        self.normalized('span')
        self.normalized('macd')

        signals = self.analyze_candlesticks()
        # print(signals)

        for index in data.index:
            row = data.iloc[index]
            print(f"{index:3d} trending {row['normalized_variance']:4d}", end=" ")
            print(f"volume {row['normalized_volume']:3d}, tension {row['normalized_tension']:4d},", end=" ")
            print(f"span {row['normalized_span']:3d} ({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end=" ")
            print(f"{row['candlestick']}", end=" ")

            if index > 5 and row['normalized_variance'] < -70 and (row['normalized_volume'] > 60 or row['normalized_volume'] < 30):
                if row['lower_wick'] > 0.2:
                    print(" *****", end="")
                    entries.append(index+1)
            elif index > 5 and row['normalized_variance'] > 70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                if row['upper_wick'] > 0:
                    print(" ----------", end="")
                    exits.append(index+1)
            print()

        positions = [0] * len(data)
        for i in entries:
            positions[i] = 1
        for i in exits:
            positions[i] = -1
        data['position'] = positions

        print(entries)
        print(exits)
        self.snapshot([0, 100], ['normalized_tension', 'normalized_volume'])

    def signal(self):
        self.raft()
