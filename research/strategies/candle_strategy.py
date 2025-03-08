from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class CandleStrategy(Strategy):

    @staticmethod
    def scenario(structure):
        position = 0
        match structure:
            case ('sudden decrease', 'momentum valley', 'gradual increase'):
                print("Buy with high confidence. Strong demand absorption and continuation.")
                position = 1

            case ('gradual increase', 'momentum peak', 'sharp pullback'):
                print("Sell with medium confidence. Possible reversal or temporary correction.")
                position = -1

            case ('rapid increase', 'buyer exhaustion', 'sudden reversal'):
                print("Sell with high confidence. Exhaustion and distribution detected.")

            case ('steady downtrend', 'false breakdown', 'strong reversal'):
                print("Buy with high confidence. Possible bottom formation.")
                position = 1

            case ('sideways range', 'fake breakout', 'return to range'):
                print("Wait or fade breakout. Market remains indecisive.")

            case ('steady uptrend', 'volume spike reversal', 'failed recovery'):
                print("Sell with medium to high confidence. Trend breakdown confirmed.")
                position = -1

            case ('rapid sell-off', 'climactic bottom', 'sharp rebound'):
                print("Buy with medium confidence. Exhaustion detected.")
                position = 1

            case ('slow grind upwards', 'supply zone rejection', 'failed breakdown'):
                print("Hold or wait. Lack of conviction in either direction.")

            case ('consolidation at highs', 'breakout', 'continuation up'):
                print("Buy with high confidence. Breakout confirmed.")
                position = 1

            case _:
                print(f"Processing default logic for {structure}")

        if 'bear trap' in structure[0] and 'reversal up' in structure[2]:
            position = 1

        if 'bull trap' in structure[0] and 'reversal down' in structure[2]:
            position = -1

        return position

    def comment(self, idx):
        data = self.data
        row = data.iloc[idx]
        price = row['close']
        upper = int(row['upper'] * 100)
        body = int(row['body'] * 100)
        lower = int(row['lower'] * 100)
        span = int(row['span'] / price * 10000)
        volume = row['volume'] // 10000

        if self.context is not None:
            volumes = self.context['volume']
            volume_base = [int(volumes.mean() / 390), int(volumes.max() / 390), int(volumes.min() / 390)]
        else:
            volume_base = [12, 15, 10]
        strong_volume = volume_base[1] // 10000
        moderate_volume = volume_base[2] // 10000

        tension = row['tension']
        macd = row['macd']
        macd_signal = row['signal_line']
        strength = row['strength']
        atr = row['atr']
        rvol = row['rvol']
        comment = []

        # ATR-Based Stop and Target
        atr_multiplier, stop_multiplier, rvol_threshold = 1.5, 0.75, 1.5
        target_price = price + atr_multiplier * atr if body > 0 else price - atr_multiplier * atr
        stop_loss = price - stop_multiplier * atr if body > 0 else price + stop_multiplier * atr
        if pd.notna(target_price) and pd.notna(stop_loss):
            if target_price > price > stop_loss:
                comment.append(f"{target_price:.2f}[{price:.2f}]{stop_loss:.2f}")

        # stop loss operations
        pass

        if abs(body) > 80 and span > 25:
            direction = "bullish" if body > 0 else "bearish"
            if volume > 20:
                comment.append(f"Strong {direction} move with high volume")
            else:
                comment.append(f"Strong {direction} move but low volume")

        # Detect trend exhaustion using normalized clustered volume
        if abs(body) < 20:  # High avg volume, but small body (absorption)
            comment.append("Possible absorption (high avg volume but little price movement)")

        # VWAP-based analysis
        if tension > 0 > body and strength > 0:
            comment.append(
                "Bullish: Price > VWAP, potential bullish continuation after a pullback")
        elif tension < 0 < body and strength < 0:
            comment.append("Bearish: Price < VWAP, potential bearish continuation after a rally")

        if abs(tension) > 25:
            comment.append("Reversal: High tension between price and VWAP, potential reversion move")

        # Resistance (long upper wick with high volume strengthens the signal)
        if strength > 0 and upper > 30 and span > 15:
            if volume > strong_volume:
                comment.append("Potential strong resistance (long upper wick, high volume)")
            elif volume > moderate_volume:
                comment.append("Potential weak resistance (long upper wick, moderate volume)")

        # Support (long lower wick with high volume confirms demand)
        if strength < 0 and lower > 25 and span > 15:
            if volume > strong_volume:
                comment.append("Potential strong support (long lower wick, high volume)")
            elif volume > moderate_volume:
                comment.append("Potential weak support (long lower wick, moderate volume)")

        # Buying Pressure (strong bullish body, rising volume, and trend shift)
        if body > 50 and volume > strong_volume and macd > 0:
            comment.append("Bullish: Buying pressure detected (body, vol, trend)")

        # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
        if body < -50 and volume > strong_volume and macd < 0:
            comment.append("Bearish: Selling pressure detected (body, vol, trend)")

        # Exit if VWAP reversion is likely
        if 0 < macd and abs(tension) > 35 and volume < moderate_volume:
            comment.append("Strong VWAP mean reversion detected, trend weakening, low volume")

        if 0 < macd and rvol < rvol_threshold and volume < moderate_volume:
            comment.append("Weak momentum (RVOL low), trend losing strength")

        if idx > 0:
            prev_row = data.iloc[idx - 1]
            prev_body = int(prev_row['body'] * 100)
            prev_volume = prev_row['volume'] // 10000
            prev_trend = prev_row['trending']
            prev_macd = prev_row['macd']
            prev_macd_signal = prev_row['signal_line']

            # Bullish reversal: Lower wick absorption, increasing volume, trend shift, VWAP confirmation, and MACD crossover
            if (
                prev_body < 0 < body
                and lower > 30 and span > 50
                and volume > prev_volume
                and prev_trend < -50 and macd > 0
                and macd > macd_signal and prev_macd <= prev_macd_signal
            ):
                comment.append("Bullish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

            # Bearish reversal: Upper wick exhaustion, increasing volume, trend shift, VWAP confirmation, and MACD crossover
            if (
                prev_body > 0 > body
                and upper > 30 and span > 50
                and volume > prev_volume
                and prev_trend > 50 and macd < 0
                and macd < macd_signal and prev_macd >= prev_macd_signal
            ):
                comment.append("Bearish reversal signal confirmed by volume, trend shift, VWAP move, and MACD crossover")

            if strength > prev_row['strength'] > 0:
                comment.append("Bullish: MACD momentum increasing")
            if strength < prev_row['strength'] < 0:
                comment.append("Bearish: MACD momentum decreasing")

        return comment

    def keypoint(self, p, structure):
        data = self.data
        key_point_signal = "neutral"

        # Consider 3 bars around p
        key_vol = data['volume'].iloc[p - 1:p + 2]
        key_price = data['close'].iloc[p - 1:p + 2]

        # Linear regression slopes
        key_vol_trend = np.polyfit(range(3), key_vol, 1)[0]  # Volume trend at key point
        key_price_trend = np.polyfit(range(3), key_price, 1)[0]  # Price trend at key point

        # Define a small threshold to avoid false classification of near-zero slopes
        slope_threshold = 1e-6  # Adjust as needed

        def slope_direction(value):
            if value > slope_threshold:
                return "up"
            elif value < -slope_threshold:
                return "down"
            else:
                return "flat"

        vol_dir = slope_direction(key_vol_trend)
        price_dir = slope_direction(key_price_trend)

        if structure == "peak":
            # Examples of refined naming
            if vol_dir == "up" and price_dir == "up":
                key_point_signal = "momentum peak"  # Price & volume both rising into the peak
            elif vol_dir == "down" and price_dir == "up":
                key_point_signal = "buyer exhaustion"  # Price up, volume fading
            elif vol_dir == "up" and price_dir == "down":
                key_point_signal = "volume spike reversal"  # Volume up, but price turning down
            else:
                key_point_signal = "calm peak"
        elif structure == "valley":
            if vol_dir == "up" and price_dir == "up":
                key_point_signal = "momentum valley"
            elif vol_dir == "down" and price_dir == "up":
                key_point_signal = "strong demand absorption"
            elif vol_dir == "up" and price_dir == "down":
                key_point_signal = "false breakdown"
            else:
                key_point_signal = "calm valley"

        return key_point_signal

    def follow_through(self, start, end):
        """
        Checks the candlestick bodies in a range to see if there's a strong follow-through move.
        Instead of returning after the first bar, we watch the entire interval.
        """
        data = self.data
        signal = "neutral"
        positive_bars = 0
        negative_bars = 0

        for i in range(start, end):
            row = data.iloc[i]
            # Assuming row['body'] is a decimal ratio * 100 => e.g. 50 = 50%
            body_ratio = row['body'] * 100

            # If body_ratio > 40 => bullish bar
            if body_ratio > 40:
                positive_bars += 1
            elif body_ratio < -40:
                negative_bars += 1

        # Simple logic: if majority are bullish => "reversal up", if majority are bearish => "reversal down"
        if positive_bars > negative_bars and positive_bars > 0:
            signal = "reversal up"
        elif negative_bars > positive_bars and negative_bars > 0:
            signal = "reversal down"

        return signal

    def cluster(self, start, end):
        """
        Summarize volume, price range, and 'combined candle' characteristics over [start, end).
        """
        data = self.data
        segment = data.iloc[start:end]

        volume = segment['volume'].sum()
        high = segment['high'].max()
        low = segment['low'].min()
        open_ = data['open'].iloc[start]
        close_ = data['close'].iloc[end - 1]

        span = high - low
        # Body is relative to the entire span in that interval
        body_ratio = ((close_ - open_) / (span + 1e-6)) * 100

        # Wicks: top/bottom as % of total range
        upper_wick_ratio = ((high - max(close_, open_)) / (span + 1e-6)) * 100
        lower_wick_ratio = ((min(close_, open_) - low) / (span + 1e-6)) * 100

        # Return scaled or integer values as needed
        volume_k = volume // 10000
        span_pct = (span / max(open_, 1e-6)) * 10000  # e.g. "range" in basis points
        # Round or convert to int
        upper_wick = int(round(upper_wick_ratio))
        body = int(round(body_ratio))
        lower_wick = int(round(lower_wick_ratio))

        return volume_k, int(span_pct), [upper_wick, body, lower_wick]

    def summarize(self, p, index, peaks, valleys):
        data = self.data
        patterns, trading_signals = [], []

        # Determine if this point is a peak or valley
        if p in peaks:
            structure = 'peak'
            start = max([x for x in valleys if x < p - 1], default=0)
            end = next((x for x in valleys if x > p + 1), index)
        elif p in valleys:
            structure = 'valley'
            start = max([x for x in peaks if x < p - 1], default=0)
            end = next((x for x in peaks if x > p + 1), index)
        else:
            structure = 'neutral'
            start, end = 0, index

        # 1) Key point signal
        key_point_signal = self.keypoint(p, structure)

        # 2) Follow-through analysis
        follow_through_signal = self.follow_through(p + 1, end + 1)

        # 3) Phases to analyze
        phases = [(start, p), (p, p + 1), (p + 1, end + 1)]

        for (ph_start, ph_end) in phases:
            cv, span, c = self.cluster(ph_start, ph_end)
            print(f"[{ph_start} - {ph_end - 1}] vol {cv:3d}, candle {span} {c} at {structure} @{p}")

            # Only proceed if there's more than 1 bar
            if ph_end - ph_start > 1:
                vol = data['volume'].iloc[ph_start:ph_end]
                vwap = data['vwap'].iloc[ph_start:ph_end]
                price = data['close'].iloc[ph_start:ph_end]

                vol_average = vol.mean()
                vol_std = vol.std()
                vol_max_min_ratio = vol.max() / max(vol.min(), 1)  # Avoid zero division

                duration = ph_end - ph_start
                vol_trend_slope = np.polyfit(range(duration), vol, 1)[0]
                price_trend_slope = np.polyfit(range(duration), price, 1)[0]
                vwap_trend_slope = np.polyfit(range(duration), vwap, 1)[0]

                price_change = price.iloc[-1] - price.iloc[0]

                # Volume Pattern Classification
                steep_slope_threshold = 0.21 * vol_average
                moderate_slope_threshold = 0.08 * vol_average

                if vol_trend_slope > moderate_slope_threshold:
                    if vol_trend_slope > steep_slope_threshold:
                        volume_pattern = "super strong increase"
                    elif vol_std / vol_average < 0.2:
                        volume_pattern = "steady increase"
                    elif vol_max_min_ratio > 2:
                        volume_pattern = "strong increase"
                    else:
                        volume_pattern = "erratic increase"
                elif vol_trend_slope < -moderate_slope_threshold:
                    if vol_trend_slope < -steep_slope_threshold:
                        volume_pattern = "super strong decrease"
                    elif vol_std / vol_average < 0.2:
                        volume_pattern = "steady decrease"
                    elif vol_max_min_ratio > 2:
                        volume_pattern = "strong decrease"
                    else:
                        volume_pattern = "erratic decrease"
                else:
                    volume_pattern = "flat or stable"

                patterns.append(volume_pattern)

                # Candle analysis from cluster
                long_upper_wick = c[0] > 60
                candle_body = c[1]  # in % of range
                long_lower_wick = c[2] > 60

                # Example: tension field > 0 => price reclaim, < 0 => price reject
                last = data.iloc[ph_end - 1]
                price_reclaim = last['tension'] > 0
                price_reject = last['tension'] < 0

                # Refined Price Direction with Threshold
                slope_threshold = 0.1  # Adjust if needed
                if price_trend_slope > slope_threshold:
                    price_direction = "up"
                elif price_trend_slope < -slope_threshold:
                    price_direction = "down"
                else:
                    price_direction = "flat"

                # Refined Volume Direction (including 'flat' possibility)
                if "increase" in volume_pattern:
                    volume_direction = "up"
                elif "decrease" in volume_pattern:
                    volume_direction = "down"
                else:
                    volume_direction = "flat"

                # --- Generate Trading Signals ---
                if price_direction == "up" and volume_direction == "up":
                    if any(x in volume_pattern for x in ["super strong", "strong"]):
                        if candle_body > 50 and price_reclaim:
                            trading_signal = "strong bullish continuation"
                        else:
                            trading_signal = "bullish continuation - caution"
                    elif "steady" in volume_pattern:
                        trading_signal = "bullish continuation - steady momentum"
                    elif "erratic" in volume_pattern:
                        trading_signal = "bullish continuation - erratic volume"
                    else:  # "flat or stable"
                        trading_signal = "bullish but volume stable"

                elif price_direction == "up" and volume_direction == "down":
                    if any(x in volume_pattern for x in ["super strong", "strong"]):
                        if candle_body < 30 and long_upper_wick and price_reject:
                            trading_signal = "bull trap confirmed - strong reversal down"
                        else:
                            trading_signal = "bull trap possible - needs more confirmation"
                    elif "steady" in volume_pattern:
                        trading_signal = "weak bullish move - watch for reversal"
                    elif "erratic" in volume_pattern:
                        trading_signal = "bullish uncertainty - caution advised"
                    else:  # "flat or stable"
                        trading_signal = "bullish but volume dropping - potential exhaustion"

                elif price_direction == "down" and volume_direction == "up":
                    if any(x in volume_pattern for x in ["super strong", "strong"]):
                        if candle_body < 30 and long_lower_wick and price_reclaim:
                            trading_signal = "bear trap confirmed - strong reversal up"
                        else:
                            trading_signal = "bear trap possible - needs more confirmation"
                    elif "steady" in volume_pattern:
                        trading_signal = "weak bearish move - watch for reversal"
                    elif "erratic" in volume_pattern:
                        trading_signal = "bearish uncertainty - caution advised"
                    else:  # "flat or stable"
                        trading_signal = "bearish but volume stable - possible absorption"

                elif price_direction == "down" and volume_direction == "down":
                    if any(x in volume_pattern for x in ["super strong", "strong"]):
                        if candle_body > 50 and price_reject:
                            trading_signal = "strong bearish continuation"
                        else:
                            trading_signal = "bearish continuation - caution"
                    elif "steady" in volume_pattern:
                        trading_signal = "bearish continuation - steady momentum"
                    elif "erratic" in volume_pattern:
                        trading_signal = "bearish continuation - erratic volume"
                    else:  # "flat or stable"
                        trading_signal = "bearish but volume stable"

                else:
                    # Covers "flat" price or volume combos
                    trading_signal = "neutral or inconclusive"

                trading_signals.append(trading_signal)

                # Debug/monitoring prints
                print(f"  Volume Pattern: {volume_pattern}, Trading Signal: {trading_signal}")
                print(f"  Vol avg/std: {int(vol_average // 10000)}/{int(vol_std // 10000)} "
                      f"Slope: {vol_trend_slope:.2f}, Max/Min: {vol_max_min_ratio:.2f}")
                print(f"  Price Trend Slope: {price_trend_slope:.2f}, VWAP Trend Slope: {vwap_trend_slope:.2f}")

        # Combine the first pattern & first signal with the key point & follow-through
        if patterns and trading_signals:
            return patterns[0] + ', ' + trading_signals[0], key_point_signal, follow_through_signal
        else:
            return "no-pattern", key_point_signal, follow_through_signal

    def spot(self, index):
        row = self.data.iloc[index]
        price = row['close']
        print(f"{index:3d} 📈{row['macd'] * 100:4.1f}", end=" ")
        print(f"🚿{row['volume'] // 10000:3d}, 🏹{int(row['tension']):4d}", end=" ")
        candle = f"🕯️{int(row['span'] / price * 10000)} ({row['upper'] * 100:.0f} {row['body'] * 100:.0f} {row['lower'] * 100:.0f})"
        print(f"{candle:18} {', '.join(self.comment(index))}")

    def candle(self):
        data = self.data
        prev_peaks, prev_valleys = set(), set()
        prev_vol_peaks, prev_vol_valleys = set(), set()
        positions = []
        for index in range(len(data)):
            position = 0
            visible_rows = data.loc[:index]
            prices, highs, lows, volumes = visible_rows['close'], visible_rows['high'], visible_rows['low'], visible_rows['volume']

            self.spot(index)

            peaks, _ = find_peaks(prices, distance=5)
            valleys, _ = find_peaks(-prices, distance=5)
            new_peaks = [p for p in peaks if p > 5 > index - p and p not in prev_peaks]
            new_valleys = [v for v in valleys if v > 5 > index - v and v not in prev_valleys]

            vol_peaks, _ = find_peaks(volumes, distance=5)
            vol_valleys, _ = find_peaks(-volumes, distance=5)
            new_vol_peaks = [p for p in vol_peaks if p > 5 > index - p and p not in prev_vol_peaks]
            new_vol_valleys = [v for v in vol_valleys if v > 5 > index - v and v not in prev_vol_valleys]

            limiter = "- " * 36
            if len(vol_peaks) and index > vol_peaks[-1] + 1 and len(new_vol_peaks):
                print(f"{limiter} vol_peaks found {new_vol_peaks} @{index}")
                prev_vol_peaks.update(vol_peaks)
            if len(vol_valleys) and index > vol_valleys[-1] + 1 and len(new_vol_valleys):
                print(f"{limiter} vol_valleys found {new_vol_valleys} @{index}")
                prev_vol_valleys.update(vol_valleys)

            if len(peaks) and index > peaks[-1] + 1 and len(new_peaks):
                print(f"{limiter} peaks found {new_peaks} @{index}")
                for p in new_peaks:
                    if p in vol_peaks or p in vol_valleys:
                        position = self.scenario(self.summarize(p, index, peaks, valleys))
                prev_peaks.update(peaks)
            if len(valleys) and index > valleys[-1] + 1 and len(new_valleys):
                print(f"{limiter} valleys found {new_valleys} @{index}")
                for v in new_valleys:
                    if v in vol_peaks or v in vol_valleys:
                        position = self.scenario(self.summarize(v, index, peaks, valleys))
                prev_valleys.update(valleys)

            positions.append(position)
        data['position'] = positions
        self.data = data
        # self.snapshot([20, 100])

    def signal(self):
        self.candle()

