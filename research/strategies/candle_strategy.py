from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


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
        distance = 3
        prominence = data.iloc[0]['close'] * 0.00125 + 0.005
        prev_peaks, prev_valleys = [], []
        positions = []
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

            trending_decision = '*' if index in collected else ' '
            print(f"{index:3d}{trending_decision} 🚀{row['normalized_trending']:4d}", end=" ")
            print(f"🌪️{row['normalized_volume']:3d}, 🧲{row['normalized_tension']:4d}", end=" ")
            candle = f"🕯️{row['normalized_span']} ({row['upper_wick'] * 100:.0f} {row['body_size'] * 100:.0f} {row['lower_wick'] * 100:.0f})"
            print(f"{candle:18}", end=" ")
            print(f"{row['candlestick']}", end=" ")
            print()

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]

            new_peaks = [p for p in peaks if p not in prev_peaks and index - p < 20]
            new_valleys = [v for v in valleys if v not in prev_valleys and index - v < 20]

            if len(new_peaks):
                print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - new found peaks {new_peaks} @{index}")
                for p in new_peaks:
                    cv, cb = ca.cluster(p - 3, p)
                    print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - clustered volume {cv} prior to peak @{p}")
                    if cv > 150 and cb > 20:
                        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                        _, cb = ca.cluster(p - 1, p + 1)
                        if cb < -20:
                            position = -1
                    else:
                        print()
                # evaluate resistance: momentum(/), demand-supply, market structure, smart money(?)
            if len(new_valleys):
                print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - new found valleys {new_valleys} @{index}")
                for v in new_valleys:
                    cv, cb = ca.cluster(v - 3, v)
                    print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - clustered volume {cv} prior to valley @{v}", end="")
                    if cv > 150 and cb < -20:
                        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                        _, cb = ca.cluster(v - 1, v + 1)
                        if cb > 20:
                            position = 1
                    else:
                        print()
            prev_peaks, prev_valleys = peaks, valleys

            positions.append(position)

        data['position'] = positions
        self.data = data

        self.snapshot([150, 240], ['strength', 'normalized_volume'])

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

        def cluster(self, start, end):
            data = self.data
            clustered_volume, clustered_body = 0, 0
            for i in range(start, end):
                prev_row, row = data.iloc[i], data.iloc[i+1]
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
                open = row['open']
                close = row['close']
                high = row['high']
                low = row['low']
                normalized_tension = row['normalized_tension']
                macd = row['macd']
                macd_signal = row['signal_line']
                strength = row['strength']
                atr = row['atr']
                rvol = row['rvol']

                clustered_volume += volume
                clustered_body += body

            return clustered_volume, clustered_body


        def follow_up(self, start, end):            
            data = self.data
            signal = []
            prev_bars = []  # Stores past bars for analysis
            rejection_count = 0  # Track consecutive bearish bars after resistance
            breakout_count = 0  # Track consecutive bullish bars after breakout
            confidence_score = 0  # Track confidence of breakout/rejection
            breakout_monitor = 0  # Monitor breakout sustainability
            breakout_failed = False  # Track failed breakouts

            for i in range(start, end):
                prev_row, row = data.iloc[i], data.iloc[i+1]
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
                open = row['open']
                close = row['close']
                high = row['high']
                low = row['low']
                normalized_tension = row['normalized_tension']
                macd = row['macd']
                macd_signal = row['signal_line']
                strength = row['strength']
                atr = row['atr']
                rvol = row['rvol']

                prev_upper = int(prev_row['upper_wick'] * 100)
                prev_body = int(prev_row['body_size'] * 100)
                prev_lower = int(prev_row['lower_wick'] * 100)
                prev_span = prev_row['normalized_span']
                prev_volume = prev_row['normalized_volume']
                prev_trend = prev_row['normalized_trending']
                prev_trend_clustered_volume = prev_row['trend_clustered_volume']
                prev_trend_clustered_volume_avg = prev_row['trend_clustered_volume_avg']
                prev_vwap = prev_row['vwap']
                prev_normalized_vwap = prev_row['normalized_vwap']
                prev_tension = prev_row['tension']
                prev_open = prev_row['open']
                prev_close = prev_row['close']
                prev_high = prev_row['high']
                prev_low = prev_row['low']
                prev_normalized_tension = prev_row['normalized_tension']
                prev_macd = prev_row['macd']
                prev_macd_signal = prev_row['signal_line']
                prev_strength = prev_row['strength']
                prev_atr = prev_row['atr']
                prev_rvol = prev_row['rvol']

                # Resistance Detection
                if prev_strength < 0 and prev_upper > 30 and abs(prev_body) > 30:
                    if prev_volume > 60:
                        signal.append(f"Bar {i}: Potential strong resistance (long upper wick, high volume)")
                    elif prev_volume > 40:
                        signal.append(f"Bar {i}: Potential weak resistance (long upper wick, moderate volume)")

                    # Check for breakout attempt
                    if close > prev_high and strength > 0:  # Require MACD confirmation
                        breakout_count += 1
                        confidence_score += volume // 20  # Increase confidence based on volume
                        breakout_monitor = 3  # Monitor for 3 bars after breakout
                    else:
                        breakout_count = 0  # Reset breakout count if price falls back
                        confidence_score = 0  # Reset confidence score

                    # Monitor breakout sustainability
                    if breakout_monitor > 0:
                        breakout_monitor -= 1
                        if close < prev_high:
                            confidence_score -= 2  # Reduce confidence if price falls back
                        if breakout_monitor == 0 and confidence_score < 5:
                            signal.append(f"Bar {i}: Breakout failed, price returned to resistance zone")

                    # Check for rejection (multi-bar confirmation)
                    if close < prev_close and body < -30 and volume > 50 and strength < prev_strength:
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
                if prev_strength > 0 and prev_lower > 30 and abs(prev_body) > 30:
                    if prev_volume > 60:
                        signal.append(f"Bar {i}: Potential strong support (long lower wick, high volume)")
                    elif prev_volume > 40:
                        signal.append(f"Bar {i}: Potential weak support (long lower wick, moderate volume)")

                    # Check for breakdown attempt
                    if close < prev_low and strength < 0:  # Require MACD confirmation
                        breakout_count += 1
                        confidence_score += volume // 20  # Increase confidence based on volume
                        breakout_monitor = 3  # Monitor for 3 bars after breakdown
                    else:
                        breakout_count = 0  # Reset breakout count if price rises back
                        confidence_score = 0  # Reset confidence score

                    # Monitor breakdown sustainability
                    if breakout_monitor > 0:
                        breakout_monitor -= 1
                        if close > prev_low:
                            confidence_score -= 2  # Reduce confidence if price returns to support
                        if breakout_monitor == 0 and confidence_score < 5:
                            signal.append(f"Bar {i}: Breakdown failed, price returned to support zone")

                    # Check for bounce confirmation (multi-bar validation)
                    if close > prev_close and body > 30 and volume > 50 and strength > prev_strength:
                        rejection_count += 1
                        confidence_score += volume // 20  # Increase confidence based on volume
                    else:
                        rejection_count = 0  # Reset count if bounce not confirmed
                        confidence_score = 0  # Reset confidence score

                    if rejection_count >= 2 and confidence_score > 5:
                        signal.append(
                            f"Bar {i}: Confirmed support bounce (strong buy signal, confidence {confidence_score})")
                    elif rejection_count >= 2:
                        signal.append(
                            f"Bar {i}: Confirmed support bounce (weak buy signal, confidence {confidence_score})")
                    rejection_count = 0  # Reset count after confirming support bounce

            return signal

        def detect_momentum(self, idx, window=10):
            data = self.data[:idx + 1]
            price = data.iloc[idx]['close']
            recent_prices = data['close'].iloc[-window:]
            recent_macd = data['macd'].iloc[-window:]
            recent_signal = data['signal_line'].iloc[-window:]
            recent_strength = data['strength'].iloc[-window:]

            # Identify MACD trend direction
            macd_trend = np.polyfit(range(window), recent_macd, 1)[0]  # Linear regression slope
            strength_trend = np.polyfit(range(window), recent_strength, 1)[0]  # Slope of MACD strength

            # Determine price behavior relative to key level
            price_near_level = abs(recent_prices.iloc[-1] - price) <= price * 0.02  # Within 2% of level

            # Conditions for breakout and breakdown
            if price_near_level and recent_macd.iloc[-1] > recent_signal.iloc[-1] and macd_trend > 0:
                return 'breakout'
            elif price_near_level and recent_macd.iloc[-1] < recent_signal.iloc[-1] and macd_trend < 0:
                return 'breakdown'

            # Detect divergence for rejection or bounce
            price_trend = np.polyfit(range(window), recent_prices, 1)[0]  # Price trend slope
            if price_near_level and price_trend > 0 > macd_trend:
                return 'rejection'
            elif price_near_level and price_trend < 0 < macd_trend:
                return 'bounce'

            return 'neutral'

        def analyze(self, idx=0):
            signals = []
            positions = []
            todos = []

            self.atr()
            self.rvol()
            self.cluster_volume()
            if idx > 5:
                data = self.data.iloc[idx-3:idx+2]
            else:
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
                    if target_price > price > stop_loss:
                        signal.append(f"{target_price:.2f}[{price:.2f}]{stop_loss:.2f}")

                # stop loss operations
                pass

                # if len(todos):
                #     num_todos = len(todos)
                #     for i in range(num_todos):
                #         t = todos[i]
                #         if 6 > idx - t[0] > 2:
                #             print(f"@{idx} to confirm {t[1]} @{t[0]}")
                #             follow_up = self.follow_up(t[0], idx)
                #             print(follow_up)

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
                if strength > 0 and upper > 30 and span > 20:
                    if volume > 60:
                        signal.append("Potential strong resistance (long upper wick, high volume)")
                        todos.append((idx, 'strong resistance'))
                    elif volume > 40:
                        signal.append("Potential weak resistance (long upper wick, moderate volume)")
                        todos.append((idx, 'weak resistance'))

                # Support (long lower wick with high volume confirms demand)
                if strength < 0 and lower > 30 and span > 20:
                    if volume > 60:
                        signal.append("Potential strong support (long lower wick, high volume)")
                        todos.append((idx, 'strong support'))
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
                    and prev_trend < -50 and trend > -10
                    and normalized_vwap > prev_vwap
                    and macd > macd_signal and prev_macd <= prev_macd_signal
                ):
                    signal.append("Bullish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")
                    position = 1

                # Bearish reversal: Upper wick exhaustion, increasing volume, trend shift, VWAP confirmation, and MACD crossover
                if (
                    prev_body > 0 > body
                    and upper > 30 and span > 50
                    and volume > prev_volume
                    and prev_trend > 50 and trend < 10
                    and normalized_vwap < prev_vwap
                    and macd < macd_signal and prev_macd >= prev_macd_signal
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
