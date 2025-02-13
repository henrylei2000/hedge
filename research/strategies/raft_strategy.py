from strategy import Strategy


class RaftStrategy(Strategy):

    def analyze_candlesticks(self, cluster_window=6):
        data = self.data
        signals = []

        # Calculate rolling sum of volume over a fixed window
        data['clustered_volume'] = data['normalized_volume'].rolling(window=cluster_window, min_periods=1).sum()

        # Calculate trend-based volume clustering (volume sum during consecutive bullish or bearish trends)
        data['trend_clustered_volume'] = 0
        data['trend_clustered_volume_avg'] = 0
        trend_volume = 0
        trend_bars = 0
        trend_direction = None

        for idx in data.index:
            row = data.loc[idx]
            body = row['body_size']
            volume = row['normalized_volume']

            if trend_direction is None or (trend_direction > 0 and body < 0) or (trend_direction < 0 and body > 0):
                trend_volume = volume  # Reset trend volume when trend changes
                trend_bars = 1
                trend_direction = 1 if body > 0 else -1
            else:
                trend_volume += volume
                trend_bars += 1

            data.at[idx, 'trend_clustered_volume'] = trend_volume
            data.at[idx, 'trend_clustered_volume_avg'] = trend_volume // trend_bars if trend_bars > 0 else 0

        for idx, row in data.iterrows():
            upper = int(row['upper_wick'] * 100)
            body = int(row['body_size'] * 100)
            lower = int(row['lower_wick'] * 100)
            span = row['normalized_span']
            volume = row['normalized_volume']
            trend = row['trending']
            clustered_volume = row['clustered_volume']
            trend_clustered_volume = row['trend_clustered_volume']
            trend_clustered_volume_avg = row['trend_clustered_volume_avg']
            vwap = row['vwap']
            normalized_vwap = row['normalized_vwap']
            tension = row['tension']
            normalized_tension = row['normalized_tension']
            macd = row['macd']  # Adding MACD data for reversal confirmation
            macd_signal = row['signal_line']
            signal = []

            # Strong move (large body with high volume confirms significance)
            if abs(body) > 80 and span > 50:
                direction = "bullish" if body > 0 else "bearish"
                if volume > 60:
                    signal.append(f"Strong {direction} move with high volume")
                else:
                    signal.append(f"Strong {direction} move but low volume")

            # Detect trend exhaustion using normalized clustered volume
            if trend_clustered_volume_avg > 50 and abs(body) < 20:  # High avg volume, but small body (absorption)
                signal.append("Possible absorption (high avg volume but little price movement)")

            # VWAP-based analysis
            if tension > 0 and body < 0:
                signal.append("Price below VWAP, bearish pressure")
            elif tension < 0 and body > 0:
                signal.append("Price above VWAP, bullish pressure")

            if abs(normalized_tension) > 50:
                signal.append("High tension between price and VWAP, potential reversion move")

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

            # Buying Pressure (strong bullish body, rising volume, and trend shift)
            if body > 50 and volume > 60 and trend > 10:
                signal.append("Buying pressure detected")

            # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
            if body < -50 and volume > 60 and trend < -10:
                signal.append("Selling pressure detected")

            # Reversal signs with VWAP and MACD integration
            if idx > 0:
                prev_row = data.iloc[idx - 1]
                prev_upper = int(prev_row['upper_wick'] * 100)
                prev_body = int(prev_row['body_size'] * 100)
                prev_lower = int(prev_row['lower_wick'] * 100)
                prev_span = prev_row['normalized_span']
                prev_volume = prev_row['normalized_volume']
                prev_trend = prev_row['trending']
                prev_vwap = prev_row['normalized_vwap']
                prev_macd = prev_row['macd']
                prev_macd_signal = prev_row['signal_line']

                # Bullish reversal: Lower wick absorption, increasing volume, trend shift, VWAP confirmation, and MACD crossover
                if prev_body < 0 < body and lower > 30 and span > 50 and volume > prev_volume and prev_trend < -50 and trend > -10 and normalized_vwap > prev_vwap and macd > macd_signal and prev_macd <= prev_macd_signal:
                    signal.append(
                        "Bullish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

                # Bearish reversal: Upper wick exhaustion, increasing volume, trend shift, VWAP confirmation, and MACD crossover
                if prev_body > 0 > body and upper > 30 and span > 50 and volume > prev_volume and prev_trend > 50 and trend < 10 and normalized_vwap < prev_vwap and macd < macd_signal and prev_macd >= prev_macd_signal:
                    signal.append(
                        "Bearish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

            signals.append((idx, ", ".join(signal) if signal else ""))

        # Assign signals back to DataFrame
        data['candlestick'] = data.index.map(dict(signals))
        return signals

    def raft(self):
        data = self.data
        data['position'] = 0
        buckets_in_use = 0
        entries, exits = [], []
        self.normalized('trending')
        self.normalized('volume')
        self.normalized('vwap')
        self.normalized('tension')
        self.normalized('span')
        self.normalized('macd')

        signals = self.analyze_candlesticks()
        # print(signals)

        for index in data.index:
            row = data.iloc[index]
            print(f"{index:3d} trending {row['normalized_trending']:4d}", end=" ")
            print(f"volume {row['normalized_volume']:3d}, tension {row['normalized_tension']:4d},", end=" ")
            print(f"span {row['normalized_span']:3d} ({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end=" ")
            print(f"{row['candlestick']}", end=" ")

            if index > 5 and row['normalized_trending'] < -70 and (row['normalized_volume'] > 60 or row['normalized_volume'] < 30):
                if row['lower_wick'] > 0.2:
                    print(" *****", end="")
                    entries.append(index+1)
            elif index > 5 and row['normalized_trending'] > 70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
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

        print(f"entries: {entries}")
        print(f"exits: {exits}")

        self.snapshot([0, 100], ['normalized_tension', 'normalized_volume'])

    def signal(self):
        self.raft()
