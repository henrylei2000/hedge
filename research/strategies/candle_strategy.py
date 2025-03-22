from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class CandleStrategy(Strategy):

    def volume_context(self):
        if self.context is not None:
            volumes = self.context['volume']
            volume_base = [int(volumes.mean() / 390), int(volumes.max() / 390), int(volumes.min() / 390)]
        else:
            volume_base = [12, 15, 10]
        typical_volume = volume_base[0] // 10000
        strong_volume = volume_base[1] // 10000
        moderate_volume = volume_base[2] // 10000
        return typical_volume, strong_volume, moderate_volume

    def comment(self, idx):
        data = self.data
        row = data.iloc[idx]
        price = row['close']
        upper = int(row['upper'] * 100)
        body = int(row['body'] * 100)
        lower = int(row['lower'] * 100)
        span = int(row['span'] / price * 10000)
        volume = row['volume'] // 10000

        _, strong_volume, moderate_volume = self.volume_context()

        tension = row['tension']
        macd = row['macd']
        macd_signal = row['macd_signal']
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
            if volume > strong_volume:
                comment.append(f"Strong {direction} move with high volume")
            elif volume > moderate_volume:
                comment.append(f"Strong {direction} move with moderate volume")
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
            prev_macd_signal = prev_row['macd_signal']

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

    def pivot_context(self, prev_p, current_p,
                      baseline_period=10,
                      use_momentum=False,
                      consecutive_vol_thresh=2):

        data = self.data

        # Ensure indices are valid
        start_idx = min(prev_p, current_p)
        end_idx = max(prev_p, current_p) + 1  # slice end is exclusive

        if start_idx < 0 or end_idx > len(data):
            return {"context_signal": "invalid range"}

        segment = data.iloc[start_idx:end_idx]
        n = len(segment)
        if n < 2:
            return {"context_signal": "not enough bars"}

        # --- 1) Price & Volume Slopes in [prev_p, current_p] ---
        price_array = segment['close']
        volume_array = segment['volume']
        segment_price_avg = price_array.mean()
        segment_vol_avg = volume_array.mean()

        price_dir, volume_dir = self.slope(segment)

        # --- 2) Volume Acceleration (compare average volume to baseline) ---
        # baseline_period bars before start_idx
        baseline_start = max(0, start_idx - baseline_period)
        baseline_end = start_idx  # slice end is exclusive
        baseline_segment = data.iloc[baseline_start:baseline_end]
        if len(baseline_segment) > 0:
            baseline_vol_avg = baseline_segment['volume'].mean()
        else:
            baseline_vol_avg = segment_vol_avg  # fallback if no data

        # Ratio of segment volume to baseline
        if baseline_vol_avg > 0:
            vol_acc_ratio = segment_vol_avg / baseline_vol_avg
        else:
            vol_acc_ratio = 1.0  # avoid division by zero

        # We'll define a threshold to consider it "accelerated volume"
        # e.g., if ratio > 1.5 => 50% higher volume
        accelerated_volume = (vol_acc_ratio > 1.5)

        # --- 3) Multi-Bar Volume Patterns (accumulation / distribution) ---
        # Check consecutive rising or falling volume
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        last_vol = None

        for i in range(start_idx, end_idx):
            current_vol = data.iloc[i]['volume']
            if last_vol is not None:
                if current_vol > last_vol:
                    consecutive_up += 1
                    consecutive_down = 0
                elif current_vol < last_vol:
                    consecutive_down += 1
                    consecutive_up = 0
                # else equal => reset both?
            last_vol = current_vol
            max_consecutive_up = max(max_consecutive_up, consecutive_up)
            max_consecutive_down = max(max_consecutive_down, consecutive_down)

        # Decide a volume_pattern label
        if max_consecutive_up >= consecutive_vol_thresh:
            volume_pattern = "steady accumulation"
        elif max_consecutive_down >= consecutive_vol_thresh:
            volume_pattern = "steady distribution"
        else:
            volume_pattern = "mixed volume"

        # If accelerated volume is detected, we can refine the label
        if accelerated_volume and "accumulation" in volume_pattern:
            volume_pattern = "accelerated accumulation"
        elif accelerated_volume and "distribution" in volume_pattern:
            volume_pattern = "accelerated distribution"
        elif accelerated_volume:
            volume_pattern = "accelerated volume"

        # --- 4) Price-Volume Divergence ---
        # If price is up but volume is strongly down => exhaustion uptrend
        # If price is down but volume is strongly up => accumulation downtrend
        # We'll define "strong" as "strong up"/"strong down" in the classification
        divergence_label = None
        if price_dir.startswith("strong up") and volume_dir.startswith("strong down"):
            divergence_label = "exhaustion uptrend"
        elif price_dir.startswith("strong down") and volume_dir.startswith("strong up"):
            divergence_label = "accumulation downtrend"

        # --- 5) (Optional) RSI / MACD usage (selective) ---
        if use_momentum:
            rsi_start = segment.iloc[0]['rsi'] if 'rsi' in segment.columns else None
            rsi_end = segment.iloc[-1]['rsi'] if 'rsi' in segment.columns else None
            rsi_label_start = self.classify_rsi(rsi_start) if rsi_start is not None else "N/A"
            rsi_label_end = self.classify_rsi(rsi_end) if rsi_end is not None else "N/A"
            rsi_trend = f"{rsi_label_start}->{rsi_label_end}"

            macd_start = segment.iloc[0]['macd'] if 'macd' in segment.columns else None
            macd_signal_start = segment.iloc[0]['macd_signal'] if 'macd_signal' in segment.columns else None
            macd_end = segment.iloc[-1]['macd'] if 'macd' in segment.columns else None
            macd_signal_end = segment.iloc[-1]['macd_signal'] if 'macd_signal' in segment.columns else None
            macd_label_start = self.classify_macd(macd_start, macd_signal_start) if macd_start is not None else "N/A"
            macd_label_end = self.classify_macd(macd_end, macd_signal_end) if macd_end is not None else "N/A"
            macd_trend = f"{macd_label_start}->{macd_label_end}"
        else:
            rsi_trend = "N/A->N/A"
            macd_trend = "N/A->N/A"

        # --- 6) High-Level Candlestick Stats (for reference) ---
        bullish_count = 0
        bearish_count = 0
        for i in range(start_idx, end_idx):
            body_ratio = data.iloc[i]['body'] * 100
            if body_ratio > 0:
                bullish_count += 1
            elif body_ratio < 0:
                bearish_count += 1

        if bullish_count > bearish_count:
            dominant_candle = "bullish"
        elif bearish_count > bullish_count:
            dominant_candle = "bearish"
        else:
            dominant_candle = "mixed"

        # --- 7) VWAP Positioning (optional) ---
        above_vwap_count = 0
        for i in range(start_idx, end_idx):
            row = data.iloc[i]
            if row['close'] > row['vwap']:
                above_vwap_count += 1
        vwap_ratio = above_vwap_count / n
        if vwap_ratio > 0.7:
            vwap_stance = "mostly above"
        elif vwap_ratio < 0.3:
            vwap_stance = "mostly below"
        else:
            vwap_stance = "mixed"

        # --- 8) Combine into a Final Context Signal ---
        # Priority:
        # 1) Check for strong price-volume divergence
        # 2) Otherwise check alignment of price_dir & volume_dir
        # 3) Incorporate volume_pattern (accumulation, distribution, etc.)
        # 4) Momentum signals only use_momentum=True (could break ties or add commentary)
        if divergence_label:
            context_signal = divergence_label
        else:
            if price_dir.endswith("up") and volume_dir.endswith("up"):
                # Possibly 'volume-driven uptrend'
                if "strong" in price_dir or "strong" in volume_dir:
                    context_signal = "volume-driven uptrend"
                else:
                    context_signal = "mild volume uptrend"
            elif price_dir.endswith("down") and volume_dir.endswith("down"):
                if "strong" in price_dir or "strong" in volume_dir:
                    context_signal = "volume-driven downtrend"
                else:
                    context_signal = "mild volume downtrend"
            else:
                context_signal = "range-bound or mixed"

        # We can refine final context based on volume_pattern
        # e.g., if context_signal is 'volume-driven uptrend' but we see 'accelerated accumulation',
        # we can combine them:
        if "uptrend" in context_signal and "accumulation" in volume_pattern:
            context_signal += " + accumulation"
        elif "downtrend" in context_signal and "distribution" in volume_pattern:
            context_signal += " + distribution"

        # If volume is accelerated, but context is mixed => "mixed + accelerated volume"
        if "mixed" in context_signal and "accelerated volume" in volume_pattern:
            context_signal = "mixed + accelerated volume"

        return {
            "observation": f"{prev_p}-{current_p}",
            "price_slope": price_dir,
            "volume_slope": volume_dir,
            "volume_pattern": volume_pattern,          # steady/accelerated accumulation/distribution, etc.
            "accelerated_volume": accelerated_volume,  # bool
            "divergence_label": divergence_label,      # exhaustion uptrend, accumulation downtrend, or None
            "rsi_trend": rsi_trend,
            "macd_trend": macd_trend,
            "dominant_candle": dominant_candle,
            "vwap_stance": vwap_stance,
            "context_signal": context_signal
        }

    @staticmethod
    def slope(segment):
        n = len(segment)
        price_array = segment['close']
        volume_array = segment['volume']
        price_slope = np.polyfit(range(n), price_array, 1)[0]
        volume_slope = np.polyfit(range(n), volume_array, 1)[0]
        segment_price_avg = price_array.mean()
        segment_vol_avg = volume_array.mean()

        price_dir = Strategy.slope_classification(price_slope / segment_price_avg, 0.001, 0.002)
        volume_dir = Strategy.slope_classification(volume_slope / segment_vol_avg, 0.05, 0.1)

        return price_dir, volume_dir

    @staticmethod
    def detect_single_candle(bar, body_threshold=0.4, wick_threshold=0.5):
        """
        Analyzes a single bar (Series with open, high, low, close) for various patterns:
          - Hammer / Inverted Hammer
          - Shooting Star
          - Doji
          - Strong bearish shooting star
          - Strong bullish hammer
        Returns a string describing the pattern or None.
        """
        o = bar['open']
        c = bar['close']
        h = bar['high']
        l = bar['low']

        candle_range = max(h - l, 1e-6)
        body_size = abs(c - o)
        upper_wick = (h - max(o, c)) / candle_range
        lower_wick = (min(o, c) - l) / candle_range
        body_ratio = body_size / candle_range

        # Detect Doji
        if body_ratio < 0.1 and upper_wick > 0.4 and lower_wick > 0.4:
            return "doji"

        # Standard Hammer/Inverted Hammer
        elif body_ratio < body_threshold and lower_wick > wick_threshold and upper_wick < 0.1:
            return "hammer" if c > o else "inverted hammer"

        # Standard Shooting Star
        elif body_ratio < body_threshold and upper_wick > wick_threshold:
            return "shooting star"

        # Strong bearish shooting star (your scenario: large bearish body, big upper wick, tiny/no lower wick)
        elif (c < o) and body_ratio >= body_threshold and upper_wick >= wick_threshold and lower_wick < 0.1:
            return "strong bearish shooting star"

        # Strong bullish hammer (for completeness, bullish candle with large lower wick)
        elif (c > o) and body_ratio >= body_threshold and lower_wick >= wick_threshold and upper_wick < 0.1:
            return "strong bullish hammer"

        return None

    @staticmethod
    def detect_three_candle_pattern(bar_prev, bar_mid, bar_next):
        """
        Analyzes three consecutive bars for patterns like:
          - Morning Star / Evening Star
          - Bullish / Bearish Engulfing
        Returns a string or None.
        """
        o1, c1 = bar_prev['open'], bar_prev['close']
        o2, c2 = bar_mid['open'], bar_mid['close']
        o3, c3 = bar_next['open'], bar_next['close']

        # Bullish Engulfing (simplified example)
        if (c2 > o2) and (c1 < o1) and (o2 < c1) and (c2 > o1):
            return "bullish engulfing"

        # Bearish Engulfing (simplified example)
        if (c2 < o2) and (c1 > o1) and (o2 > c1) and (c2 < o1):
            return "bearish engulfing"

        # Morning Star (simplified)
        if (c1 < o1) and abs(c1 - o1) > abs(o1) * 0.005:
            if abs(c2 - o2) < abs(c1 - o1) * 0.5:
                if c3 > o3 and (c3 - o3) > abs(c1 - o1) * 0.5:
                    return "morning star"

        # Evening Star (simplified)
        if (c1 > o1) and abs(c1 - o1) > abs(o1) * 0.005:
            if abs(c2 - o2) < abs(c1 - o1) * 0.5:
                if c3 < o3 and (o3 - c3) > abs(c1 - o1) * 0.5:
                    return "evening star"

        return None

    def keypoint(self, p, structure, half_window=1, use_rsi=True, use_macd=True, confirm_bars=2):
        """
        Analyzes [p - half_window, p + half_window] to classify a peak or valley,
        incorporates candlestick patterns (single-bar & 3-bar) directly into expected_dir,
        optionally uses RSI/MACD, and finally calls follow_through() for confirmation.

        Returns: (key_point_signal, follow_through_signal)
        """
        data = self.data
        key_point_signal = "neutral"
        expected_dir = None

        start_idx = max(0, p - half_window)
        end_idx = min(len(data), p + half_window + 1)
        segment = data.iloc[start_idx:end_idx]

        # Edge case: if segment too small, return neutral
        if len(segment) < 2:
            return key_point_signal, "neutral"

        # 1) Evaluate local slopes
        price_dir, volume_dir = self.slope(segment)

        # 2) Check local max/min
        middle_close = data.iloc[p]['close']
        highest_close = segment['close'].max()
        lowest_close = segment['close'].min()
        is_local_peak = (middle_close == highest_close)
        is_local_valley = (middle_close == lowest_close)

        # 3) Basic pivot logic
        if structure == "peak":
            expected_dir = "down"
            if is_local_peak:
                if price_dir.startswith("strong up") and volume_dir.startswith("strong up"):
                    key_point_signal = "momentum peak"
                elif price_dir.endswith("up") and volume_dir.startswith("strong down"):
                    key_point_signal = "buyer exhaustion"
                elif price_dir.startswith("strong down") and volume_dir.startswith("strong up"):
                    key_point_signal = "volume spike reversal"
                else:
                    key_point_signal = "calm peak"
            else:
                key_point_signal = "soft peak - not local max"

        elif structure == "valley":
            expected_dir = "up"
            if is_local_valley:
                if price_dir.startswith("strong up") and volume_dir.startswith("strong up"):
                    key_point_signal = "momentum valley"
                elif price_dir.endswith("up") and volume_dir.startswith("strong down"):
                    key_point_signal = "strong demand absorption"
                elif price_dir.startswith("strong down") and volume_dir.startswith("strong up"):
                    key_point_signal = "false breakdown"
                else:
                    key_point_signal = "calm valley"
            else:
                key_point_signal = "soft valley - not local min"

        # 4) Candlestick Analysis
        # A) Single-bar (middle bar p)
        bar_p = data.iloc[p][['open', 'high', 'low', 'close']]
        single_pattern = self.detect_single_candle(bar_p)
        if single_pattern:
            key_point_signal += f" + {single_pattern}"

        # B) 3-bar pattern if segment has 3 bars
        if len(segment) == 3:
            bar_prev = data.iloc[p - 1][['open', 'high', 'low', 'close']]
            bar_mid = data.iloc[p][['open', 'high', 'low', 'close']]
            bar_next = data.iloc[p + 1][['open', 'high', 'low', 'close']]
            three_bar_pattern = self.detect_three_candle_pattern(bar_prev, bar_mid, bar_next)
            if three_bar_pattern:
                key_point_signal += f" + {three_bar_pattern}"

        # 5) RSI / MACD
        rsi_label = "N/A"
        macd_label = "N/A"
        if use_rsi and 'rsi' in segment.columns:
            avg_rsi = segment['rsi'].mean()
            rsi_label = Strategy.classify_rsi(avg_rsi)

        if use_macd and 'macd' in segment.columns and 'macd_signal' in segment.columns:
            avg_macd = segment['macd'].mean()
            avg_macd_signal = segment['macd_signal'].mean()
            macd_label = Strategy.classify_macd(avg_macd, avg_macd_signal)

        # Adjust key_point_signal based on RSI/MACD
        if structure == "peak" and rsi_label == "overbought":
            key_point_signal += " + RSI overbought"
        elif structure == "valley" and rsi_label == "oversold":
            key_point_signal += " + RSI oversold"

        if use_macd:
            if "peak" in structure and macd_label == "bearish":
                key_point_signal += " + MACD Bearish"
            elif "valley" in structure and macd_label == "bullish":
                key_point_signal += " + MACD Bullish"

        bullish_patterns = ["hammer", "morning star", "bullish engulfing", "inverted hammer"]  # or others
        bearish_patterns = ["shooting star", "evening star", "bearish engulfing"]

        if expected_dir:
            follow_start = p + 1
            follow_end = min(len(data), p + 1 + confirm_bars)
            follow_signal = self.follow_through(follow_start, follow_end, expected_direction=expected_dir)
        else:
            follow_signal = "neutral"

        return key_point_signal, follow_signal

    @staticmethod
    def keypoint_direction(key_point_signal):
        """
        Map the key point signal to an expected follow-through direction.
        """
        if key_point_signal in ["momentum valley", "strong demand absorption"]:
            return "up"
        elif key_point_signal in ["momentum peak", "buyer exhaustion", "volume spike reversal"]:
            return "down"
        else:
            return None  # e.g. "calm valley", "calm peak" => no strong bias

    def follow_through(self, start, end, expected_direction=None):
        data = self.data
        if end - start < 1:
            return "neutral"

        last_bar = data.iloc[end - 1]
        self.detect_single_candle(last_bar)
        segment = data.iloc[start:end]
        price_dir, volume_dir = self.slope(segment)
        actual_direction = "mixed"
        if price_dir.endswith("up"):
            actual_direction = "up"
        elif price_dir.endswith("down"):
            actual_direction = "down"
        if expected_direction == actual_direction:
            return "confirmed " + actual_direction
        else:
            return f"no follow-through (expected {expected_direction} got {actual_direction})"

    def analyze_pivot(self, prev_p, current_p, structure='peak', half_window=2):
        # 1) Macro context from prev_p to current_p
        context = self.pivot_context(prev_p, current_p)

        # 2) Micro (local) analysis around the current pivot
        kp_signal, ft_signal = self.keypoint(p=current_p, structure=structure, half_window=half_window)

        # 3) Generate a final trading decision (text-based)
        decision_dict = self._generate_trading_decision(context, kp_signal, ft_signal, structure)

        # 4) Convert that text-based recommendation into a continuous score
        score = self._score_recommendation_text(decision_dict["decision_text"])

        # 5) Return everything
        return {
            "macro_context": context,
            "micro_signal": kp_signal,
            "follow_through": ft_signal,
            "trading_decision": {
                "decision_text": decision_dict["decision_text"],
                "score": score
            }
        }

    @staticmethod
    def _generate_trading_decision(context, kp_signal, ft_signal, structure):
        """
        Combines macro context (from pivot_context), micro keypoint signal,
        and follow-through signal to produce a final trading recommendation.
        Places a heavier emphasis on follow-through: if it's not 'confirmed up/down',
        we strongly discourage any trade.
        """

        # --- 1) Extract macro context ---
        macro_signal = context.get("context_signal", "mixed")
        divergence_label = context.get("divergence_label", None)
        volume_pattern = context.get("volume_pattern", "mixed volume")
        accelerated_vol = context.get("accelerated_volume", False)

        # RSI / MACD / VWAP
        macro_rsi = context.get("rsi_trend", "neutral->neutral")
        macro_macd = context.get("macd_trend", "neutral->neutral")
        macro_vwap = context.get("vwap_stance", "mixed")

        # --- 2) Extract micro signals ---
        micro_kp = kp_signal  # e.g. 'momentum valley', 'buyer exhaustion', etc.
        micro_ft = ft_signal  # e.g. 'confirmed up', 'confirmed down', or 'no follow-through (mixed)'

        recommendation = "Hold / No Clear Trade"

        # --- 3) If there's a divergence_label, it overrides the normal macro logic ---
        if divergence_label == "exhaustion uptrend":
            recommendation = "Possible Uptrend Exhaustion - Watch for Bearish Reversal"
        elif divergence_label == "accumulation downtrend":
            recommendation = "Potential Accumulation in Downtrend - Watch for Bullish Reversal"

        # Helper to quickly check if we do NOT have a valid follow-through
        def no_follow_through():
            return ("confirmed up" not in micro_ft) and ("confirmed down" not in micro_ft)

        # --- 4) Standard Macro Logic, but we require confirmation ---
        if "uptrend" in macro_signal and divergence_label is None:
            # Macro is bullish
            if structure == "valley":
                if any(x in micro_kp for x in ["momentum valley", "strong demand absorption"]):
                    if "confirmed up" in micro_ft:
                        recommendation = "Go Long (Bullish Reversal Confirmed)"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                elif "false breakdown" in micro_kp:
                    if "confirmed up" in micro_ft:
                        recommendation = "Bear Trap -> Go Long"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                else:
                    # If it's just a weak local valley
                    if no_follow_through():
                        recommendation = "No Follow-Through => Avoid Trade"
                    else:
                        recommendation = "Potential Buy on Dip (Macro Uptrend) but Weak Local Signal"

            else:
                # structure == 'peak'
                if "momentum peak" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Short-Term Pullback, but Long-Term Uptrend"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                elif "buyer exhaustion" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Possible Reversal, Go Short"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                else:
                    if no_follow_through():
                        recommendation = "No Follow-Through => Avoid Trade"
                    else:
                        recommendation = "Peak in Uptrend - Potential Minor Correction"

        elif "downtrend" in macro_signal and divergence_label is None:
            # Macro is bearish
            if structure == "peak":
                if any(x in micro_kp for x in ["momentum peak", "buyer exhaustion"]):
                    if "confirmed down" in micro_ft:
                        recommendation = "Go Short (Bearish Continuation Confirmed)"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                elif "volume spike reversal" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Volume Spike -> Short Entry Confirmed"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                else:
                    if no_follow_through():
                        recommendation = "No Follow-Through => Avoid Trade"
                    else:
                        recommendation = "Peak in Downtrend - Potential Sell Rally"
            else:
                # structure == 'valley'
                if "momentum valley" in micro_kp:
                    if no_follow_through():
                        recommendation = "No Follow-Through => Avoid Trade"
                    else:
                        recommendation = "Potential Bearish Retracement, Wait for Confirmation"
                elif "false breakdown" in micro_kp:
                    if "confirmed up" in micro_ft:
                        recommendation = "Bullish Divergence, Possible Short Squeeze"
                    else:
                        recommendation = "No Follow-Through => Avoid Trade"
                else:
                    if no_follow_through():
                        recommendation = "No Follow-Through => Avoid Trade"
                    else:
                        recommendation = "Weak Valley in Downtrend, Could Break Lower"

        elif divergence_label is None:
            # Macro is range-bound or mixed
            if "confirmed up" in micro_ft:
                recommendation = "Range-Bound but Micro Up -> Possible Quick Long"
            elif "confirmed down" in micro_ft:
                recommendation = "Range-Bound but Micro Down -> Possible Quick Short"
            else:
                recommendation = "No Follow-Through => Avoid Trade"

        # --- 5) Volume Pattern & Acceleration Enhancements ---
        volume_pattern = context.get("volume_pattern", "mixed volume")
        accelerated_vol = context.get("accelerated_volume", False)
        if "accumulation" in volume_pattern.lower():
            recommendation += " | Volume Accumulation => Bullish Lean"
        elif "distribution" in volume_pattern.lower():
            recommendation += " | Volume Distribution => Bearish Lean"
        if accelerated_vol:
            recommendation += " | Accelerated Volume => Stronger Conviction"

        # --- 6) RSI / MACD extremes ---
        macro_rsi = context.get("rsi_trend", "neutral->neutral")
        macro_macd = context.get("macd_trend", "neutral->neutral")
        rsi_end = macro_rsi.split("->")[-1] if "->" in macro_rsi else "neutral"
        macd_end = macro_macd.split("->")[-1] if "->" in macro_macd else "neutral"

        if "peak" in structure and "overbought" in rsi_end:
            recommendation += " | RSI Overbought => Strengthens Bearish Bias"
        elif "valley" in structure and "oversold" in rsi_end:
            recommendation += " | RSI Oversold => Strengthens Bullish Bias"

        if "peak" in structure and "bearish" in macd_end.lower():
            recommendation += " | MACD Bearish => Confirms Potential Reversal"
        elif "valley" in structure and "bullish" in macd_end.lower():
            recommendation += " | MACD Bullish => Confirms Potential Reversal"

        # --- 7) VWAP stance ---
        macro_vwap = context.get("vwap_stance", "mixed")
        if "above" in macro_vwap:
            recommendation += " | Price Mostly Above VWAP => Bullish Lean"
        elif "below" in macro_vwap:
            recommendation += " | Price Mostly Below VWAP => Bearish Lean"

        # --- 8) Candlestick Patterns from `kp_signal` ---
        # We'll scan for known candlestick keywords in the micro keypoint signal
        candlestick_map = {
            "hammer": "Hammer => Bullish Lean",
            "inverted hammer": "Inverted Hammer => Potential Bullish Reversal",
            "morning star": "Morning Star => Bullish Reversal",
            "bullish engulfing": "Bullish Engulfing => Strong Bullish Reversal",
            "shooting star": "Shooting Star => Bearish Reversal",
            "evening star": "Evening Star => Bearish Reversal",
            "bearish engulfing": "Bearish Engulfing => Strong Bearish Reversal",
            "doji": "Doji => Indecision/Reversal",
            "strong bearish shooting star": "Strong Bearish Shooting Star => Strong Bearish Reversal",
            "strong bullish hammer": "Strong Bullish Hammer => Strong Bullish Reversal"
        }

        kp_lower = micro_kp.lower()
        for pattern, text in candlestick_map.items():
            if pattern in kp_lower:
                recommendation += f" | {text}"

        ft_lower = micro_ft.lower()
        for pattern, text in candlestick_map.items():
            if pattern in ft_lower:
                recommendation += f" | {text}"

        return {"decision_text": recommendation}

    @staticmethod
    def _score_recommendation_text(recommendation):
        """
        Converts a human-readable recommendation string into a continuous score in [-1, 1].
          +1 => strong buy
          +0.5 => mild buy
           0  => hold
          -0.5 => mild sell
          -1  => strong sell

        Incorporates volume/divergence references, momentum signals, VWAP stance,
        AND candlestick patterns as additive modifiers.
        """
        rec_lower = recommendation.lower()

        # --- 1) Base Keywords -> Base Score ---
        base_map = {
            "go long": 1.0,
            "buy": 1.0,
            "bullish reversal confirmed": 0.9,
            "quick long": 0.8,
            "partial profit": 0.2,  # might be less bullish if we talk about profit-taking
            "go short": -1.0,
            "sell": -1.0,
            "bearish continuation confirmed": -0.9,
            "quick short": -0.8,
            "pullback": -0.3,
            "minor correction": -0.2
        }

        # --- 2) Multipliers for Uncertainty Phrases ---
        #    Scale the absolute value of the base score
        multiplier_map = {
            "possible": 0.6,
            "watch": 0.7,
            "caution": 0.7,
            "weak": 0.5
        }

        # --- 3) Additive Modifiers for Volume/Divergence/RSI/MACD/VWAP *and* Candlestick Patterns ---
        #    We *add* these values to the final score if the phrase appears in the text.
        #    You can adjust these values to fit your strategy.
        additive_map = {
            # Volume & Divergence signals
            "exhaustion uptrend": -0.5,
            "accumulation downtrend": +0.5,
            "volume accumulation => bullish lean": +0.3,
            "volume distribution => bearish lean": -0.3,
            "accelerated volume => stronger conviction": +0.2,

            # RSI references
            "rsi overbought => strengthens bearish bias": -0.2,
            "rsi oversold => strengthens bullish bias": +0.2,

            # MACD references
            "macd bearish => confirms potential reversal": -0.2,
            "macd bullish => confirms potential reversal": +0.2,

            # VWAP stance references
            "price mostly above vwap => bullish lean": +0.1,
            "price mostly below vwap => bearish lean": -0.1,

            # Candlestick patterns
            "hammer => bullish lean": +0.2,
            "inverted hammer => potential bullish reversal": +0.1,
            "morning star => bullish reversal": +0.3,
            "bullish engulfing => strong bullish reversal": +0.4,
            "shooting star => bearish reversal": -0.3,
            "evening star => bearish reversal": -0.4,
            "bearish engulfing => strong bearish reversal": -0.4,
            "doji => indecision/reversal": 0.0,
            "strong bearish shooting star => strong bearish reversal": -0.5,
            "strong bullish hammer => strong bullish reversal": +0.5
        }

        # --- Step A: Identify the strongest base keyword ---
        base_score = 0.0
        for phrase, val in base_map.items():
            if phrase in rec_lower:
                # Pick the phrase with the largest absolute value
                if abs(val) > abs(base_score):
                    base_score = val

        # --- Step B: Apply multipliers if present ---
        final_score = base_score
        for phrase, mult in multiplier_map.items():
            if phrase in rec_lower:
                # Scale the absolute value by the multiplier
                final_score = final_score * mult

        # --- Step C: Apply additive modifiers (sum them) ---
        # If multiple additive phrases appear, we sum them all
        for phrase, delta in additive_map.items():
            if phrase in rec_lower:
                final_score += delta

        # --- Step D: Clamp to [-1, 1] ---
        final_score = max(min(final_score, 1.0), -1.0)

        return round(final_score, 2)

    def detect_swings(self, idx=0, left=3, right=3, min_wick_pct=30, volume_threshold=None):
        """
        Advanced swing detection with conditions:
        - Minimum wick percentage
        - Volume filters
        - VWAP proximity (optional)
        """
        data = self.data[:idx]
        peaks, valleys = [], []
        volume_threshold = 0  # self.context['volume'] // 390)
        for i in range(left, len(data) - right):
            high = data['high'][i]
            low = data['low'][i]
            open_ = data['open'][i]
            close = data['close'][i]
            volume = data['volume'][i]
            vwap = data['vwap'][i]

            # Calculate wick percentages
            upper_wick = (high - max(open_, close)) / (high - low + 1e-8) * 100
            lower_wick = (min(open_, close) - low) / (high - low + 1e-8) * 100

            # Peak Condition
            peak_cond = (
                high == max(data['high'][i - left:i + right + 1]) and
                upper_wick >= min_wick_pct and
                high > vwap and
                (volume_threshold is None or volume >= volume_threshold)
            )

            if peak_cond:
                peaks.append(i)

            # Valley Condition
            valley_cond = (
                low == min(data['low'][i - left:i + right + 1]) and
                lower_wick >= min_wick_pct and
                low < vwap and
                (volume_threshold is None or volume >= volume_threshold)
            )

            if valley_cond:
                valleys.append(i)

        return peaks, valleys

    def key_level(self, idx=0):
        bar = self.data.iloc[idx]
        visible_rows = self.data.iloc[:idx]
        prices, highs, lows, volumes = visible_rows['close'], visible_rows['high'], visible_rows['low'], visible_rows['volume']
        a_p, b_p = np.polyfit(np.arange(len(highs)), highs, 1)
        resistance = a_p * idx + b_p
        a_v, b_v = np.polyfit(np.arange(len(lows)), lows, 1)
        support = a_v * idx + b_v
        print(f"R {resistance:.3f} = {a_p:.3f} * {idx} + {b_p:.3f}")
        print(f"P {bar['open']:.3f} - {bar['high']:.3f} - {bar['low']:.3f} - {bar['close']:.3f}")
        print(f"S {support:.3f} = {a_v:.3f} * {idx} + {b_v:.3f}")
        return resistance, support

    def cluster(self, start, end):
        data = self.data
        segment = data.loc[start:end]

        volume = segment['volume'].sum()
        high = segment['high'].max()
        low = segment['low'].min()
        open_ = data['open'].iloc[start]
        close_ = data['close'].iloc[end]

        span = high - low
        # Body is relative to the entire span in that interval
        body_ratio = ((close_ - open_) / (span + 1e-6)) * 100

        # Wicks: top/bottom as % of total range
        upper_wick_ratio = ((high - max(close_, open_)) / (span + 1e-6)) * 100
        lower_wick_ratio = ((min(close_, open_) - low) / (span + 1e-6)) * 100

        # Return scaled or integer values as needed
        volume_k = volume // 10000
        span_pct = (span / max(close_, 1e-6)) * 10000  # e.g. "range" in basis points
        # Round or convert to int
        upper_wick = int(round(upper_wick_ratio))
        body = int(round(body_ratio))
        lower_wick = int(round(lower_wick_ratio))

        cluster_candle = f"[{start} - {end}] cluster candle: {int(span_pct)} ({upper_wick}, {body}, {lower_wick}) - vol {volume // 10000}"
        print(cluster_candle)

        return volume, int(span_pct), [upper_wick, body, lower_wick]

    def spot(self, index):
        row = self.data.iloc[index]
        candle = self.detect_single_candle(row)
        if candle is not None:
            print(f"{index}: {candle}")
        price = row['close']
        print(f"{index:3d} 📈{row['macd'] * 100:4.1f}", end=" ")
        print(f"🚿{row['volume'] // 10000:3d}, 🏹{int(row['tension']):4d}", end=" ")
        candle = f"🕯️{int(row['span'] / price * 10000)} ({row['upper'] * 100:.0f} {row['body'] * 100:.0f} {row['lower'] * 100:.0f})"
        print(f"{candle:18} {', '.join(self.comment(index))}")
        if index > 9:
            self.cluster(max(0, index - 14), index - 10)
        if index > 4:
            self.cluster(max(0, index - 9), index - 5)
        self.cluster(max(0, index - 4), index)


    def candle(self):
        data = self.data
        positions = []
        for idx in range(len(data)):
            bar = data.iloc[idx]
            position = 0
            visible_rows = data.iloc[:idx]
            prices, highs, lows, volumes = visible_rows['close'], visible_rows['high'], visible_rows['low'], visible_rows['volume']
            peaks, _ = find_peaks(highs)
            valleys, _ = find_peaks(-lows)
            # peaks, valleys = self.detect_swings(idx)
            # p_prices, v_prices = prices.iloc[peaks], prices.iloc[valleys]

            if len(peaks) > 1 and len(valleys) > 1:
                prev_peak = peaks[-2]
                current_peak = peaks[-1]
                prev_valley = valleys[-2]
                current_valley = valleys[-1]
                # resistance, support = self.key_level(idx)

                self.spot(idx)
                if current_peak < current_valley:
                    context = self.analyze_pivot(prev_peak, current_valley, structure='valley')
                    score = context['trading_decision']['score']
                    decision = context['trading_decision']['decision_text']
                    print(f"[{score}] for {valleys[-1]}@{idx} {decision} - new found valleys {valleys[-2:]}")
                    if score > 0.5:
                        position = 1
                if current_peak > current_valley:
                    context = self.analyze_pivot(prev_valley, current_peak, structure='peak')
                    score = context['trading_decision']['score']
                    decision = context['trading_decision']['decision_text']
                    print(f"[{score}] for {peaks[-1]}@{idx} {decision}- new found peaks {peaks[-2:]}")
                    if score < -0.5:
                        position = -1

                # if current_peak == current_valley:
                #     print(f"{current_peak} == {current_valley} @ {idx} \n {peaks}\n {valleys}")

            positions.append(position)
        data['position'] = positions
        self.snapshot([30, 90], ['rvol', 'tension'])

    def dual_frame(self):
        data = self.data
        data5 = self.data5
        positions = []
        for idx in range(len(data)):
            position = 0
            visible_rows = data.iloc[:idx]
            prices, highs, lows, volumes = visible_rows['close'], visible_rows['high'], visible_rows['low'], visible_rows['volume']
            group = idx // 5
            if group > 3:
                visible_group = group - 1
                a_p, b_p = np.polyfit(np.arange(3), highs[-3:], 1)
                resistance = a_p * idx + b_p
                a_v, b_v = np.polyfit(np.arange(3), lows[-3:], 1)
                support = a_v * idx + b_v
                visible_groups = data5.iloc[:visible_group]
                group_closes = visible_groups['close']
                ga_v, gb_v = np.polyfit(np.arange(3), group_closes[-3:], 1)
                if ga_v > 0 > a_v:
                    position = 1
                if ga_v < 0 < a_p:
                    position = -1
            positions.append(position)
        data['position'] = positions

    @staticmethod
    def closest_stepstones(stepstones, b):
        smaller = [t for t in stepstones if t[1] < b]
        larger = [t for t in stepstones if t[1] > b]

        a = max(smaller, key=lambda x: x[1]) if smaller else None
        c = min(larger, key=lambda x: x[1]) if larger else None

        return a, c

    def stepstone(self):
        data = self.data
        data5 = self.data5
        stepstones = [(0, data.iloc[0]['close'])]
        positions = [0]
        typical_volume, strong_volume, moderate_volume = self.volume_context()
        for idx in range(1, len(data)):
            position = 0
            row = data.iloc[idx - 1]
            price = row['close']
            print(f"@{idx}")
            a, c = self.closest_stepstones(stepstones, price)
            if a is not None and c is not None:
                print(f"{a} --{((price - a[1]) / price * 1000):.2f}-- ({idx-1}, {price:.3f}) --{((c[1] - price) / price * 1000):.2f}-- {c}")
            elif a is not None:
                print(f"{a} --{((price - a[1]) / price * 1000):.2f}-- ({idx - 1}, {price:.3f}) < {c}")
            elif c is not None:
                print(f"{a} < ({idx - 1}, {price:.3f}) --{((c[1] - price) / price * 1000):.2f}-- {c}")
            else:
                print(f"{a} < ({idx - 1}, {price:.3f}) < {c}")
            self.spot(idx - 1)
            print()
            positions.append(position)
            if row['rvol'] > 1.25 and row['volume'] // 10000 > typical_volume:
                stepstones.append((idx, round(row['close'], 3)))
            elif row['volume'] // 10000 > 3 * strong_volume and idx > 1:
                stepstones.append((idx - 1, price, 3))
        data['position'] = positions
        self.snapshot([0, 70], ['rvol', 'tension'])

    def signal(self):
        self.stepstone()
