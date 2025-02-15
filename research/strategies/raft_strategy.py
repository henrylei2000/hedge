from strategy import Strategy
import pandas as pd
import numpy as np

class RaftStrategy(Strategy):

    def raft(self):
        buckets_in_use = 0
        entries, exits = [], []
        self.normalized('trending')
        self.normalized('volume')
        self.normalized('vwap')
        self.normalized('tension')
        self.normalized('span')
        self.normalized('macd')

        data = self.CandlestickAnalyzer(self.data).analyze()
        total, collected = self.CandlestickAnalyzer(self.data).trend_stream()
        print(total, data['normalized_trending'].sum(), collected)

        positions = [0] * len(data)

        for index in data.iloc[5:].index:
            row = data.iloc[index]
            trending_decision = '*' if index in collected else ' '
            print(f"{index:3d}{trending_decision} trending {row['normalized_trending']:4d}", end=" ")
            print(f"volume {row['normalized_volume']:3d}, tension {row['normalized_tension']:4d},", end=" ")
            print(f"span {row['normalized_span']:3d} ({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end=" ")
            print(f"{row['candlestick']}", end=" ")

            if index > 5 and row['normalized_trending'] < -70 and (row['normalized_volume'] > 60 or row['normalized_volume'] < 30):
                if row['lower_wick'] > 0.2:
                    print(" *****", end="")
                    # entries.append(index + 1)
            elif index > 5 and row['normalized_trending'] > 70 and (
                    row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                if row['upper_wick'] > 0:
                    print(" ----------", end="")
                    # exits.append(index + 1)
            print()


            for i in entries:
                positions[i] = 1
            for i in exits:
                positions[i] = -1

            if '1 Bullish' in row['candlestick']:
                entries.append(index + 1)
            if '-1 Bearish' in row['candlestick']:
                exits.append(index + 1)

        self.data['position'] = positions

        print(f"entries ({len(entries)}): {entries}")
        print(f"exits ({len(exits)}): {exits}")

        self.snapshot([50, 100], ['normalized_macd', 'normalized_volume'])

    def signal(self):
        self.raft()

    class CandlestickAnalyzer:
        def __init__(self, data, cluster_window=6, atr_multiplier=1.5, stop_multiplier=0.75, rvol_threshold=1.5):
            self.data = data.copy()
            self.cluster_window = cluster_window
            self.atr_multiplier = atr_multiplier
            self.stop_multiplier = stop_multiplier
            self.rvol_threshold = rvol_threshold

        def calculate_atr(self, period=6):
            data = self.data
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=period).mean()
            return data

        def calculate_rvol(self, period=50):
            data = self.data
            avg_volume = data['volume'].rolling(window=period).mean()
            data['rvol'] = data['volume'] / avg_volume
            return data

        def trend_stream(self, rolling_window=5, trend_sensitivity=1):

            numbers = self.data['normalized_trending']
            # Ensure input is a NumPy array for consistency
            if isinstance(numbers, pd.Series):
                numbers = numbers.to_numpy()

            collected_sum = 0
            collected_numbers = []
            collecting = False  # Start in "wait mode"
            uptrend_counter = 0  # Tracks consecutive rising numbers
            downtrend_counter = 0  # Tracks consecutive declines

            for i in range(rolling_window, len(numbers)):
                num = numbers[i]

                # Only collect if the number is positive and collecting mode is on
                if collecting:
                    collected_sum += num
                    collected_numbers.append(i)

                if num > 80:
                    collecting = True
                    continue
                if num < -80:
                    collecting = False
                    continue

                recent_trend = numbers[max(0, i - rolling_window + 1): i + 1]

                if len(recent_trend) > 1:
                    trend_diff = pd.Series(recent_trend).diff().dropna().to_numpy()
                    avg_trend = trend_diff.mean()  # Rolling trend calculation
                else:
                    avg_trend = 0

                if avg_trend > trend_sensitivity:
                    collecting = True

                if avg_trend < -trend_sensitivity and num < -avg_trend * 2:
                    collecting = False  # Stop collecting

            return collected_sum, collected_numbers

        def analyze(self):
            signals = []
            data = self.data
            # Calculate rolling sum of volume over a fixed window
            data['clustered_volume'] = data['normalized_volume'].rolling(window=self.cluster_window,
                                                                                   min_periods=1).sum()
            # Calculate trend-based volume clustering
            data['trend_clustered_volume'] = 0
            data['trend_clustered_volume_avg'] = 0
            trend_volume = 0
            trend_bars = 0
            trend_direction = None

            for idx in data.index:
                row = self.data.loc[idx]
                body = row['body_size']
                volume = row['normalized_volume']

                if trend_direction is None or (trend_direction > 0 and body < 0) or (trend_direction < 0 and body > 0):
                    trend_volume = volume
                    trend_bars = 1
                    trend_direction = 1 if body > 0 else -1
                else:
                    trend_volume += volume
                    trend_bars += 1

                data.at[idx, 'trend_clustered_volume'] = trend_volume
                data.at[idx, 'trend_clustered_volume_avg'] = trend_volume / trend_bars if trend_bars > 0 else 0

            self.calculate_atr()
            self.calculate_rvol()

            for idx, row in data.iloc[3:].iterrows():
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
                price = row['close']
                normalized_tension = row['normalized_tension']
                macd = row['macd']
                macd_signal = row['signal_line']
                atr = row['atr']
                rvol = row['rvol']
                signal = []

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
                if tension > 0 > body:
                    signal.append(
                        "Price above VWAP, potential bullish continuation after a pullback")
                elif tension < 0 < body:
                    signal.append("Price below VWAP, potential bearish continuation after a rally")

                if abs(normalized_tension) > 70:
                    signal.append("High tension between price and VWAP, potential reversion move")

                # Resistance (long upper wick with high volume strengthens the signal)
                if upper > 30 and span > 30:
                    if volume > 60:
                        signal.append("Potential strong resistance (long upper wick, high volume)")
                    elif volume > 40:
                        signal.append("Potential weak resistance (long upper wick, moderate volume)")

                # Support (long lower wick with high volume confirms demand)
                if lower > 30 and span > 30:
                    if volume > 60:
                        signal.append("Potential strong support (long lower wick, high volume)")
                    elif volume > 40:
                        signal.append("Potential weak support (long lower wick, moderate volume)")

                # Buying Pressure (strong bullish body, rising volume, and trend shift)
                if body > 50 and volume > 60 and trend > 10:
                    signal.append("Buying pressure detected")

                # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
                if body < -50 and volume > 60 and trend < -10:
                    signal.append("Selling pressure detected")

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
                if (
                    prev_body < 0 < body
                    and lower > 30 and span > 50
                    and volume > prev_volume
                    # and prev_trend < -50 and trend > -10
                    # and normalized_vwap > prev_vwap
                    # and macd > macd_signal and prev_macd <= prev_macd_signal
                ):
                    signal.append("1 Bullish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

                # Bearish reversal: Upper wick exhaustion, increasing volume, trend shift, VWAP confirmation, and MACD crossover
                if (
                    prev_body > 0 > body
                    and upper > 30 and span > 50
                    and volume > prev_volume
                    # and prev_trend > 50 and trend < 10
                    # and normalized_vwap < prev_vwap
                    # and macd < macd_signal and prev_macd >= prev_macd_signal
                ):
                    signal.append("-1 Bearish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

                if row['macd'] - row['signal_line'] > prev_row['macd'] - prev_row['signal_line']:
                    signal.append("Entry signal: MACD momentum increasing (early confirmation)")

                # ATR-Based Stop and Target
                entry_price = row['close']
                target_price = entry_price + self.atr_multiplier * atr if body > 0 else entry_price - self.atr_multiplier * atr
                stop_loss = entry_price - self.stop_multiplier * atr if body > 0 else entry_price + self.stop_multiplier * atr
                if pd.notna(target_price) and pd.notna(stop_loss):
                    signal.append(f"Current: {price:.2f}, Target: {target_price:.2f}, Stop: {stop_loss:.2f} (ATR-based)")

                # Exit if VWAP reversion is likely
                if abs(normalized_tension) > 50 and volume < 40 and trend < 10:
                    signal.append("Exit signal: Strong VWAP mean reversion detected, trend weakening, low volume")
                if rvol < self.rvol_threshold and trend < 10:
                    signal.append("Exit signal: Weak momentum (RVOL low), trend losing strength")

                signals.append((idx, ", ".join(signal) if signal else ""))

            self.data['candlestick'] = self.data.index.map(dict(signals))
            return self.data

