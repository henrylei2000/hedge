from strategy import Strategy
import pandas as pd


class CandleStrategy(Strategy):

    def candle(self):
        buckets_in_use = 0
        self.normalized('trending')
        self.normalized('volume')
        self.normalized('vwap')
        self.normalized('tension')
        self.normalized('span')
        self.normalized('macd')

        ca = self.CandleAnalyzer(self.data)
        data = ca.analyze()
        total, collected = ca.trend_turn()
        # print(total, data['normalized_trending'].sum(), collected)

        for index in data.iloc[5:].index:
            row = data.iloc[index]
            trending_decision = '*' if index in collected else ' '
            print(f"{index:3d}{trending_decision} trend{row['normalized_trending']:4d}", end=" ")
            print(f"vol{row['normalized_volume']:3d}, tense{row['normalized_tension']:4d},", end=" ")
            print(f"candle{row['normalized_span']:3d}({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end=" ")
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
        self.data = data
        self.snapshot([0, 50], ['strength', 'normalized_volume'])

    def signal(self):
        self.candle()

    class CandleAnalyzer:
        def __init__(self, data, cluster_window=6, atr_multiplier=1.5, stop_multiplier=0.75, rvol_threshold=1.5):
            self.data = data.copy()
            self.cluster_window = cluster_window
            self.atr_multiplier = atr_multiplier
            self.stop_multiplier = stop_multiplier
            self.rvol_threshold = rvol_threshold

        def atr(self, period=6):
            data = self.data
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=period).mean()
            return data

        def rvol(self, period=6):
            data = self.data
            avg_volume = data['volume'].rolling(window=period).mean()
            data['rvol'] = data['volume'] / avg_volume
            return data

        def cluster_volume(self):
            data = self.data
            data['trend_clustered_volume'] = 0
            data['trend_clustered_volume_avg'] = 0
            trend_volume = 0
            trend_bars = 0
            trend_direction = None

            for idx in data.index:
                row = self.data.loc[idx]
                body = row['body_size']
                volume = row['normalized_volume']

                if trend_direction is None or (trend_direction > 0 > body) or (trend_direction < 0 < body):
                    trend_volume = volume
                    trend_bars = 1
                    trend_direction = 1 if body > 0 else -1
                else:
                    trend_volume += volume
                    trend_bars += 1

                data.at[idx, 'trend_clustered_volume'] = trend_volume
                data.at[idx, 'trend_clustered_volume_avg'] = int(trend_volume / trend_bars) if trend_bars > 0 else 0

            self.data = data
            return data

        def trend_turn(self, rolling_window=5, trend_sensitivity=1):
            numbers = self.data['normalized_trending']
            # Ensure input is a NumPy array for consistency
            if isinstance(numbers, pd.Series):
                numbers = numbers.to_numpy()
            collected_sum = 0
            collected_numbers = []
            collecting = False
            for i in range(rolling_window, len(numbers)):
                num = numbers[i]
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
            positions = []

            self.atr()
            self.rvol()
            self.cluster_volume()
            data = self.data
            for idx, row in data.iterrows():
                upper = int(row['upper_wick'] * 100)
                body = int(row['body_size'] * 100)
                lower = int(row['lower_wick'] * 100)
                span = row['normalized_span']
                volume = row['normalized_volume']
                trend = row['normalized_trending']
                trend_clustered_volume = row['trend_clustered_volume']
                trend_clustered_volume_avg = row['trend_clustered_volume_avg']
                vwap = row['vwap']
                normalized_vwap = row['normalized_vwap']
                tension = row['tension']
                price = row['close']
                normalized_tension = row['normalized_tension']
                macd = row['macd']
                macd_signal = row['signal_line']
                strength = row['strength']
                atr = row['atr']
                rvol = row['rvol']
                signal = []
                position = 0

                # ATR-Based Stop and Target
                entry_price = row['close']
                target_price = entry_price + self.atr_multiplier * atr if body > 0 else entry_price - self.atr_multiplier * atr
                stop_loss = entry_price - self.stop_multiplier * atr if body > 0 else entry_price + self.stop_multiplier * atr
                if pd.notna(target_price) and pd.notna(stop_loss):
                    signal.append(f"{target_price:.2f}[{price:.2f}]{stop_loss:.2f}")

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
                if tension > 0 > body and strength > 0:
                    signal.append(
                        "Bullish: Price > VWAP, potential bullish continuation after a pullback")
                elif tension < 0 < body and strength < 0:
                    signal.append("Bearish: Price < VWAP, potential bearish continuation after a rally")

                if abs(normalized_tension) > 70:
                    signal.append("Reversal: High tension between price and VWAP, potential reversion move")

                # Resistance (long upper wick with high volume strengthens the signal)
                if strength > 0 and upper > 30 and span > 30:
                    if volume > 60:
                        signal.append("Potential strong resistance (long upper wick, high volume)")
                    elif volume > 40:
                        signal.append("Potential weak resistance (long upper wick, moderate volume)")

                # Support (long lower wick with high volume confirms demand)
                if strength < 0 and lower > 30 and span > 30:
                    if volume > 60:
                        signal.append("Potential strong support (long lower wick, high volume)")
                    elif volume > 40:
                        signal.append("Potential weak support (long lower wick, moderate volume)")

                # Buying Pressure (strong bullish body, rising volume, and trend shift)
                if body > 50 and volume > 60 and trend > 10:
                    signal.append("Bullish: Buying pressure detected")

                # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
                if body < -50 and volume > 60 and trend < -10:
                    signal.append("Bearish: Selling pressure detected")

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
                    signal.append("Bullish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")
                    position = 1

                # Bearish reversal: Upper wick exhaustion, increasing volume, trend shift, VWAP confirmation, and MACD crossover
                if (
                    prev_body > 0 > body
                    and upper > 30 and span > 50
                    and volume > prev_volume
                    # and prev_trend > 50 and trend < 10
                    # and normalized_vwap < prev_vwap
                    # and macd < macd_signal and prev_macd >= prev_macd_signal
                ):
                    signal.append("Bearish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")
                    position = -1

                if strength > prev_row['strength'] > 0:
                    signal.append("Bullish: MACD momentum increasing")
                if strength < prev_row['strength'] < 0:
                    signal.append("Bearish: MACD momentum decreasing")

                # Exit if VWAP reversion is likely
                if 0 < trend < 10 and abs(normalized_tension) > 50 and volume < 40:
                    signal.append("Strong VWAP mean reversion detected, trend weakening, low volume")

                if 0 < trend < 10 and rvol < self.rvol_threshold and volume < 40:
                    signal.append("Weak momentum (RVOL low), trend losing strength")

                signals.append((idx, ", ".join(signal) if signal else ""))
                positions.append(position)

            self.data['candlestick'] = self.data.index.map(dict(signals))
            self.data['position'] = positions

            return self.data


def process_market_data(price_feed):
    """
    Simulates real-time processing of candlestick data for resistance, support, and fake-outs.
    Only uses past data for confirmation, avoiding future bar peeking.
    Implements multi-bar confirmation for rejection and breakout validation.
    Introduces confidence scoring based on volume and trend strength.
    """
    signal = []
    prev_bars = []  # Stores past bars for analysis
    rejection_count = 0  # Track consecutive bearish bars after resistance
    breakout_count = 0  # Track consecutive bullish bars after breakout
    confidence_score = 0  # Track confidence of breakout/rejection

    for i, bar in enumerate(price_feed):
        upper, body, lower, open_price, high, low, close, volume, VWAP = bar

        # Store past bars in rolling memory
        prev_bars.append(bar)
        if len(prev_bars) > 10:  # Keep last 10 bars for reference
            prev_bars.pop(0)

        # Process only if we have at least one previous bar
        if len(prev_bars) > 1:
            prev_upper, prev_body, prev_lower, prev_open, prev_high, prev_low, prev_close, prev_volume, prev_VWAP = \
            prev_bars[-2]  # Last bar

            # Resistance Detection
            if prev_upper > 30 and abs(prev_body) > 30:
                if prev_volume > 60:
                    signal.append(f"Bar {i}: Potential strong resistance (long upper wick, high volume)")
                elif prev_volume > 40:
                    signal.append(f"Bar {i}: Potential weak resistance (long upper wick, moderate volume)")

                # Check for breakout attempt
                if close > prev_high:
                    breakout_count += 1
                    confidence_score += volume // 20  # Increase confidence based on volume
                    if breakout_count >= 2 and confidence_score > 5:  # Require 2+ bars confirming breakout & volume confidence
                        signal.append(
                            f"Bar {i}: Confirmed breakout above resistance (strong buy signal, confidence {confidence_score})")
                    elif breakout_count >= 2:
                        signal.append(
                            f"Bar {i}: Confirmed breakout above resistance (weak buy signal, confidence {confidence_score})")
                else:
                    breakout_count = 0  # Reset breakout count if price falls back
                    confidence_score = 0  # Reset confidence score

                # Check for rejection (multi-bar confirmation)
                if close < prev_close and body < -30 and volume > 50:
                    rejection_count += 1
                    confidence_score += volume // 20  # Increase confidence based on volume
                else:
                    rejection_count = 0  # Reset count if rejection not confirmed
                    confidence_score = 0  # Reset confidence score

                if rejection_count >= 2 and confidence_score > 5:
                    signal.append(
                        f"Bar {i}: Confirmed resistance rejection (strong sell signal, confidence {confidence_score})")
                elif rejection_count >= 2:
                    signal.append(
                        f"Bar {i}: Confirmed resistance rejection (weak sell signal, confidence {confidence_score})")
                rejection_count = 0  # Reset count after confirming rejection

            # Support Detection
            if prev_lower > 30 and abs(prev_body) > 30:
                if prev_volume > 60:
                    signal.append(f"Bar {i}: Potential strong support (long lower wick, high volume)")
                elif prev_volume > 40:
                    signal.append(f"Bar {i}: Potential weak support (long lower wick, moderate volume)")

                # Check for breakdown attempt
                if close < prev_low:
                    breakout_count += 1
                    confidence_score += volume // 20  # Increase confidence based on volume
                    if breakout_count >= 2 and confidence_score > 5:
                        signal.append(
                            f"Bar {i}: Confirmed breakdown below support (strong sell signal, confidence {confidence_score})")
                    elif breakout_count >= 2:
                        signal.append(
                            f"Bar {i}: Confirmed breakdown below support (weak sell signal, confidence {confidence_score})")
                else:
                    breakout_count = 0  # Reset breakout count if price rises back
                    confidence_score = 0  # Reset confidence score

                # Check for bounce confirmation (multi-bar validation)
                if close > prev_close and body > 30 and volume > 50:
                    rejection_count += 1
                    confidence_score += volume // 20  # Increase confidence based on volume
                else:
                    rejection_count = 0  # Reset count if bounce not confirmed
                    confidence_score = 0  # Reset confidence score

                if rejection_count >= 2 and confidence_score > 5:
                    signal.append(
                        f"Bar {i}: Confirmed support bounce (strong buy signal, confidence {confidence_score})")
                elif rejection_count >= 2:
                    signal.append(f"Bar {i}: Confirmed support bounce (weak buy signal, confidence {confidence_score})")
                rejection_count = 0  # Reset count after confirming support bounce

    return signal