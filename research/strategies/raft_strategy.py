from strategy import Strategy


class RaftStrategy(Strategy):

    def identify_trend(self, period=3):
        """
        Step 1: Identify Market Regime
            Signal: Normalized Variance + Moving Window Analysis
                Trending Market → variance consistent values above 0 (positive) or below 0 (negative)
                Mean Reversion → variance oscillates between [-20, 20] without a clear bias, the market is likely ranging.
        """
        data = self.data
        data['bullish_count'] = data['normalized_variance'].rolling(window=period).apply(lambda x: (x > 20).sum(), raw=True)
        data['bearish_count'] = data['normalized_variance'].rolling(window=period).apply(lambda x: (x < -20).sum(), raw=True)
        data['tension_sum'] = data['normalized_tension'].rolling(window=period).sum()
        data['trend'] = 0
        threshold = int(period * 2 / 3)
        data.loc[((data['bullish_count'] >= threshold) & (data['tension_sum'] > 10)), 'trend'] = 1
        data.loc[((data['bearish_count'] >= threshold) & (data['tension_sum'] < -10)), 'trend'] = -1

    def detect_breakout(self, period=8):
        """
        Detects sharp VWAP crossings (breakouts or reversals) based on:
        - Price crossing VWAP sharply
        - Normalized variance (momentum confirmation)
        - Normalized volume (institutional participation)
        - Candlestick structure (strong close, minimal wick)
        - Follow-through validation
        """

        pass


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
        self.identify_trend()


        for index in data.index:
            row = data.iloc[index]
            print(f"{index:3d} trending {row['normalized_variance']:4d}, rolling at {row['trend']:2d}", end=" ")
            print(f"volume {row['normalized_volume']:3d}, tension {row['normalized_tension']:4d}, ", end="")
            print(f"span {row['normalized_span']:3d} ({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end="")

            if index > 5 and row['normalized_variance'] < -70 and (row['normalized_volume'] > 60 or row['normalized_volume'] < 30):
                if row['lower_wick'] > 0.25:
                    print(" *****", end="")
                    entries.append(index+1)
            elif index > 5 and row['normalized_variance'] > 70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                if row['upper_wick'] > 0:
                    print(" ----------", end="")
                    exits.append(index+1)
            print()

        positions = [0] * len(data)
        first_1s = data.index[(data['trend'] == 1) & (data['trend'].shift(1) != 1)].tolist()
        first_0s = data.index[(data['trend'] == 0) & (data['trend'].shift(1) == -1)].tolist()

        for i in entries:
            positions[i] = 1
        for i in exits:
            positions[i] = -1
        data['position'] = positions

        print(entries)
        print(exits)
        self.snapshot([40, 150], ['normalized_tension', 'normalized_volume'])

    def signal(self):
        self.raft()
