from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import itertools


class CandleStrategy(Strategy):

    def interpret(self, point, index, peaks, valleys):
        pre_point, at_point, post_point = '', '', ''
        ca = self.CandleAnalyzer(self)
        ca.atr()
        ca.rvol()
        ca.cluster_volume()
        p = point
        peak = self.data.iloc[p]
        # evaluate resistance: momentum(/), demand-supply, market structure, smart money(?)

        low = max([x for x in valleys if x < p], default=0)
        high = next((x for x in valleys if x > p), index)
        cv0, c0, trend_v0 = ca.cluster(low, p)
        cv, c, trend_v = ca.cluster(p, p + 1)
        cv1, c1, trend_v1 = ca.cluster(p + 1, high + 1)
        print(f"{low} - {p} - {high}")
        print(f"🛬vol {cv0:3d}, trend_v {trend_v0:3d}, candle {c0} before peak @{p}")
        print(f"🔴vol {cv:3d}, trend_v {trend_v:3d}, candle {c} at peak @{p}")
        print(f"🛫vol {cv1:3d}, trend_v {trend_v1:3d}, candle {c1} after peak @{p}")
        # evaluate resistance: momentum(/), demand-supply, market structure, smart money(?)
        # scenario: pre-peak, peak, post-peak
        # low, low, low -> weak reversal, weak breakout, indecision
        # price increasing + volume decreasing + smaller candles: weak uptrend, likely rejection
        if trend_v0 < 21 or cv0 < 30:
            pre_point = 'low'
        if cv < 60 or peak['normalized_span'] < 50:
            at_point = 'low'
        if cv1 < 40 or cv1 < cv0:
            post_point = 'low'
            """
            Narrow price range near resistance
            Multiple touches of resistance with no conviction.
            VWAP is flat and price oscillates around it
            """
        elif 20 < trend_v0 < 61:
            pass
        elif 60 < trend_v0:
            pass

        return pre_point, at_point, post_point

    @staticmethod
    def scenario(v):
        match v:
            case ("low", "low", "low"):
                print(f"low, low, low -> weak reversal, weak breakout, indecision")
            case ("low", "moderate", "high"):
                # Unique business logic for this scenario
                print("Processing special (low, moderate, high)")
            # Add other unique cases as needed
            case _:
                # Default business logic for all other combinations
                print(f"Processing default logic for {v}")

    def candle(self):
        buckets_in_use = 0
        self.normalized('trending')
        self.normalized('volume')
        self.normalized('vwap')
        self.normalized('tension')
        self.normalized('span')
        self.normalized('macd')

        ca = self.CandleAnalyzer(self)
        data = ca.analyze()
        total, collected = ca.trend_turn()
        # print(total, data['normalized_trending'].sum(), collected)
        distance = 8
        prominence = data.iloc[0]['close'] * 0.0015
        prev_peaks, prev_valleys = set(), set()
        positions = []
        for index in range(len(data)):
            row = data.iloc[index]
            position = 0
            price, rsi, macd, strength = row['close'], row['rsi'], row['macd'], row['strength']
            visible_rows = data.loc[:index]
            prices = visible_rows['close']
            volumes = visible_rows['volume']
            macds = visible_rows['macd']
            signals = visible_rows['signal_line']
            rsis = visible_rows['rsi']
            adlines = visible_rows['a/d']

            # comment row by row
            trending_decision = '*' if index in collected else ' '
            print(f"{index:3d}{trending_decision} 📈{row['normalized_trending']:4d}", end=" ")
            print(f"🚿{row['normalized_volume']:3d}, 🏹{row['normalized_tension']:4d}", end=" ")
            candle = f"🕯️{row['normalized_span']} ({row['upper_wick'] * 100:.0f} {row['body_size'] * 100:.0f} {row['lower_wick'] * 100:.0f})"
            print(f"{candle:18} {row['candlestick']}")

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            peak_indices = np.array(peaks)
            peak_prices = prices.iloc[peaks]
            valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            valley_indices = np.array(valleys)
            valley_prices = prices.iloc[valleys]
            new_peaks = [p for p in peaks if p > distance and p not in prev_peaks and index - p < 20]
            new_valleys = [v for v in valleys if v > distance and v not in prev_valleys and index - v < 20]
            limiter = "- " * 36
            if len(new_peaks):
                print(f"{limiter} peaks found {new_peaks} @{index}")
                for p in new_peaks:
                    self.scenario(self.interpret(p, index, peaks, valleys))
                    cv, c, trend_v = ca.cluster(p - 1, p)
                    cv1, c1, trend_v1 = ca.cluster(p - 1, index)
                    if cv > 30 and cv1 > 40:
                        print("🪂" * 3)
                        position = -1
                    print(limiter)

            if len(new_valleys):
                print(f"{limiter} valleys found {new_valleys} @{index}")
                for v in new_valleys:
                    # evaluate resistance: momentum(/), demand-supply, market structure, smart money(?)
                    cv0, c0, trend_v0 = ca.cluster(v - 3, v)
                    cv, c, trend_v = ca.cluster(v, v + 1)
                    cv1, c1, trend_v1 = ca.cluster(v + 1, index + 1)
                    print(f"🛬vol {cv0:3d}, trend_v {trend_v0:3d}, candle {c0} before valley @{v}")
                    print(f"🟢vol {cv:3d}, trend_v {trend_v:3d}, candle {c} at valley @{v}")
                    print(f"🛫vol {cv1:3d}, trend_v {trend_v1:3d}, candle {c1} after valley @{v}")

                    """ 
                    Pre-Valley: 📉 📉 📈  (Moderate selling)
                    Valley: 📊 📊 📊 📉 (Fake breakdown, liquidity grab, wick)
                    Post-Valley: 📉 📈  📈 📈 (Volume increases on reversal)
                    """
                    if 70 > trend_v0 + trend_v > 20 and 30 < cv0 < 60 and cv > 40 and cv1 > 50 :
                        print(f"moderate, keep, increase -> reversal")
                        print("🏄‍♀️" * 3)
                        position = 1
                    print(limiter)
            prev_peaks.update(peaks)
            prev_valleys.update(valleys)

            positions.append(position)

        data['position'] = positions
        self.data = data

        self.snapshot([20, 90], ['normalized_tension', 'normalized_volume'])

    def signal(self):
        self.candle()

    class CandleAnalyzer:
        def __init__(self, parent, cluster_window=6, atr_multiplier=1.5, stop_multiplier=0.75, rvol_threshold=1.5):
            self.data = parent.data.copy()
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

        def get_cluster_info(self, start, end, label):
            vol, candle, trend_vol = self.cluster(start, end)
            return f"{label} vol {vol:3d}, trend_v {trend_vol:3d}, candle {candle}"

        def cluster(self, start, end):
            data = self.data
            data['normalized_volume_diff'] = data['normalized_volume'].diff().fillna(0).astype(int)
            cluster_volume, cluster_volume_diff = 0, 0
            cluster_upper, cluster_body, cluster_lower = 0, 0, 0
            # evaluate resistance: momentum(/), demand-supply, market structure, smart money(?)
            for i in range(start, end):
                row = data.iloc[i]
                upper = int(row['upper_wick'] * 100)
                body = int(row['body_size'] * 100)
                lower = int(row['lower_wick'] * 100)
                span = row['normalized_span']
                volume = row['normalized_volume']
                volume_diff = row['normalized_volume_diff']
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

                cluster_volume += volume
                cluster_upper += upper
                cluster_lower += lower
                cluster_body += body
                cluster_volume_diff += volume_diff

            return cluster_volume // (end - start), [cluster_upper, cluster_body, cluster_lower], cluster_volume_diff // (end - start)

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
