from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class CandleStrategy(Strategy):

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

    def pivot_context(self, prev_p, current_p):
        """
        Analyze the 'macro' window from [prev_p, current_p] to derive an overall trend context.

        Returns a dictionary or a string describing the market context, e.g.:
            {
              'price_slope': 'strong up',
              'volume_slope': 'mild up',
              'rsi_trend': 'neutral->overbought',
              'macd_trend': 'bullish->neutral',
              'dominant_candle': 'bullish',
              'context_signal': 'strong uptrend overall'
            }
        """
        data = self.data

        # Ensure indices are valid
        start_idx = min(prev_p, current_p)
        end_idx = max(prev_p, current_p) + 1  # slice end is exclusive

        # Safety check
        if start_idx < 0 or end_idx > len(data):
            return {"context_signal": "invalid range"}

        segment = data.iloc[start_idx:end_idx]
        n = len(segment)
        if n < 2:
            return {"context_signal": "not enough bars"}

        # --- 1) Price & Volume Slopes ---
        price_array = segment['close']
        volume_array = segment['volume']
        price_slope = np.polyfit(range(n), price_array, 1)[0]
        volume_slope = np.polyfit(range(n), volume_array, 1)[0]
        price_dir = Strategy.slope_classification(price_slope)
        volume_dir = Strategy.slope_classification(volume_slope)

        # --- 2) RSI / MACD Over the Range ---
        rsi_start = segment.iloc[0]['rsi'] if 'rsi' in segment.columns else None
        rsi_end = segment.iloc[-1]['rsi'] if 'rsi' in segment.columns else None
        rsi_label_start = Strategy.classify_rsi(rsi_start) if rsi_start is not None else "N/A"
        rsi_label_end = Strategy.classify_rsi(rsi_end) if rsi_end is not None else "N/A"
        rsi_trend = f"{rsi_label_start}->{rsi_label_end}"

        macd_start = segment.iloc[0]['macd'] if 'macd' in segment.columns else None
        macd_signal_start = segment.iloc[0]['macd_signal'] if 'macd_signal' in segment.columns else None
        macd_end = segment.iloc[-1]['macd'] if 'macd' in segment.columns else None
        macd_signal_end = segment.iloc[-1]['macd_signal'] if 'macd_signal' in segment.columns else None

        if macd_start is not None and macd_signal_start is not None:
            macd_label_start = Strategy.classify_macd(macd_start, macd_signal_start)
        else:
            macd_label_start = "N/A"

        if macd_end is not None and macd_signal_end is not None:
            macd_label_end = Strategy.classify_macd(macd_end, macd_signal_end)
        else:
            macd_label_end = "N/A"

        macd_trend = f"{macd_label_start}->{macd_label_end}"

        # --- 3) High-Level Candlestick Stats ---
        # Count bullish/bearish bars (based on 'body')
        bullish_count = 0
        bearish_count = 0
        for i in range(start_idx, end_idx):
            body_ratio = self.data.iloc[i]['body'] * 100
            if body_ratio > 0:
                bullish_count += 1
            elif body_ratio < 0:
                bearish_count += 1

        # Simple classification: which is dominant?
        if bullish_count > bearish_count:
            dominant_candle = "bullish"
        elif bearish_count > bullish_count:
            dominant_candle = "bearish"
        else:
            dominant_candle = "mixed"

        # --- 4) VWAP Positioning (optional) ---
        # e.g., how many bars close above VWAP?
        above_vwap_count = 0
        for i in range(start_idx, end_idx):
            row = self.data.iloc[i]
            if row['close'] > row['vwap']:
                above_vwap_count += 1
        vwap_ratio = above_vwap_count / n
        if vwap_ratio > 0.7:
            vwap_stance = "mostly above"
        elif vwap_ratio < 0.3:
            vwap_stance = "mostly below"
        else:
            vwap_stance = "mixed"

        # --- 5) Combine into a Context Signal ---
        # For example, if price_dir is "strong up" and dominant_candle is "bullish" => "strong uptrend overall"
        # This is just one simplistic logic approach
        if price_dir.endswith("up") and dominant_candle == "bullish":
            context_signal = "strong uptrend overall" if "strong" in price_dir else "mild uptrend overall"
        elif price_dir.endswith("down") and dominant_candle == "bearish":
            context_signal = "strong downtrend overall" if "strong" in price_dir else "mild downtrend overall"
        else:
            context_signal = "range-bound or mixed"

        return {
            "price_slope": price_dir,
            "volume_slope": volume_dir,
            "rsi_trend": rsi_trend,
            "macd_trend": macd_trend,
            "dominant_candle": dominant_candle,
            "vwap_stance": vwap_stance,
            "context_signal": context_signal
        }

    def keypoint(self, p, structure, half_window=1, use_rsi=True, use_macd=True, confirm_bars=2):
        """
        Analyzes [p - half_window, p + half_window] to classify a peak or valley.
        Then optionally uses RSI/MACD for additional context.
        Finally, calls follow_through() on the next `confirm_bars` to confirm direction.

        :param p: The index of the pivot bar
        :param structure: 'peak' or 'valley'
        :param half_window: 1 => 3 bars total, 2 => 5 bars total, etc.
        :param use_rsi: whether to incorporate RSI classification
        :param use_macd: whether to incorporate MACD classification
        :param confirm_bars: how many bars after p to check for follow-through
        :return: key_point_signal, follow_through_signal
        """
        data = self.data
        key_point_signal = "neutral"

        start_idx = max(0, p - half_window)
        end_idx = min(len(data), p + half_window + 1)  # slice end is exclusive
        segment = data.iloc[start_idx:end_idx]

        # Edge case: if segment too small, return neutral
        if len(segment) < 2:
            return key_point_signal, "neutral"

        # Price & volume arrays
        n = len(segment)
        price_ = segment['close']
        volume_ = segment['volume']

        # Slopes
        price_slope = np.polyfit(range(n), price_, 1)[0]
        volume_slope = np.polyfit(range(n), volume_, 1)[0]
        price_dir = Strategy.slope_classification(price_slope)
        vol_dir = Strategy.slope_classification(volume_slope)

        # Check if bar p is truly local max/min in that window
        middle_close = data.iloc[p]['close']
        highest_close = price_.max()
        lowest_close = price_.min()
        is_local_peak = (middle_close == highest_close)
        is_local_valley = (middle_close == lowest_close)

        # Basic pivot logic
        if structure == "peak":
            if is_local_peak:
                if price_dir.startswith("strong up") and vol_dir.startswith("strong up"):
                    key_point_signal = "momentum peak"
                elif price_dir.endswith("up") and vol_dir.startswith("strong down"):
                    key_point_signal = "buyer exhaustion"
                elif price_dir.startswith("strong down") and vol_dir.startswith("strong up"):
                    key_point_signal = "volume spike reversal"
                else:
                    key_point_signal = "calm peak"
            else:
                key_point_signal = "soft peak - not local max"

        elif structure == "valley":
            if is_local_valley:
                if price_dir.startswith("strong up") and vol_dir.startswith("strong up"):
                    key_point_signal = "momentum valley"
                elif price_dir.endswith("up") and vol_dir.startswith("strong down"):
                    key_point_signal = "strong demand absorption"
                elif price_dir.startswith("strong down") and vol_dir.startswith("strong up"):
                    key_point_signal = "false breakdown"
                else:
                    key_point_signal = "calm valley"
            else:
                key_point_signal = "soft valley - not local min"

        # --- Incorporate RSI / MACD if available ---
        rsi_label = "N/A"
        macd_label = "N/A"
        if use_rsi and 'rsi' in segment.columns:
            # e.g. average RSI in the window
            avg_rsi = segment['rsi'].mean()
            rsi_label = Strategy.classify_rsi(avg_rsi)

        if use_macd and 'macd' in segment.columns and 'macd_signal' in segment.columns:
            avg_macd = segment['macd'].mean()
            avg_macd_signal = segment['macd_signal'].mean()
            macd_label = Strategy.classify_macd(avg_macd, avg_macd_signal)

        # Adjust key_point_signal based on RSI/MACD
        # For instance, if we have a "peak" but RSI is not overbought, maybe reduce confidence
        if structure == "peak" and rsi_label == "overbought":
            key_point_signal += " + RSI overbought"
        elif structure == "valley" and rsi_label == "oversold":
            key_point_signal += " + RSI oversold"

        if use_macd:
            if "peak" in structure and macd_label == "bearish":
                key_point_signal += " + MACD Bearish"
            elif "valley" in structure and macd_label == "bullish":
                key_point_signal += " + MACD Bullish"

        # Decide expected direction from key_point_signal
        # e.g., "momentum valley" => we expect a bullish follow-through
        if any(x in key_point_signal for x in ["valley", "demand absorption", "false breakdown"]):
            expected_dir = "up"
        elif any(x in key_point_signal for x in ["peak", "exhaustion", "reversal"]):
            expected_dir = "down"
        else:
            expected_dir = None  # no strong bias

        # --- Follow-Through Check ---
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
        """
        Checks multiple factors over [start, end) to confirm if the market
        actually follows the expected direction ('up' or 'down').
        """
        data = self.data
        if end - start < 1:
            return "neutral"

        segment = data.iloc[start:end]
        n_bars = len(segment)

        # Price slope
        price_slope = np.polyfit(range(n_bars), segment['close'], 1)[0]
        price_dir = Strategy.slope_classification(price_slope)

        # Volume slope
        volume_slope = np.polyfit(range(n_bars), segment['volume'], 1)[0]
        volume_dir = Strategy.slope_classification(volume_slope)

        # Count bullish/bearish bars by body ratio
        bullish_bars = 0
        bearish_bars = 0
        for i in range(start, end):
            body_ratio = data.iloc[i]['body'] * 100
            if body_ratio > 40:
                bullish_bars += 1
            elif body_ratio < -40:
                bearish_bars += 1

        # VWAP check: how many bars close above VWAP?
        above_vwap_count = 0
        for i in range(start, end):
            row = data.iloc[i]
            if row['close'] > row['vwap']:
                above_vwap_count += 1
        vwap_bullish = (above_vwap_count / n_bars) > 0.6
        vwap_bearish = (above_vwap_count / n_bars) < 0.4

        # Decide the overall direction in [start, end)
        # If price_dir is "strong up" or "mild up", volume_dir is up,
        # majority bars are bullish, and VWAP is bullish => "up" bias.
        total_strong_bars = bullish_bars + bearish_bars
        bull_ratio = bullish_bars / total_strong_bars if total_strong_bars > 0 else 0
        bear_ratio = bearish_bars / total_strong_bars if total_strong_bars > 0 else 0

        def direction_bias():
            if (price_dir.endswith("up") and volume_dir.endswith("up")
                    and bull_ratio > 0.6 and vwap_bullish):
                return "up"
            elif (price_dir.endswith("down") and volume_dir.endswith("down")
                  and bear_ratio > 0.6 and vwap_bearish):
                return "down"
            else:
                return "mixed"

        actual_direction = direction_bias()

        # Compare actual direction with the expected direction from the key point
        if expected_direction == "up" and actual_direction == "up":
            return "confirmed up"
        elif expected_direction == "down" and actual_direction == "down":
            return "confirmed down"
        else:
            return f"no follow-through ({actual_direction})"

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

    def analyze_pivot(self, prev_p, current_p, structure='peak', half_window=2):
        """
        Performs macro + micro analysis around a pivot and provides a trading recommendation
        as a continuous score in [-1, 1].
        """
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
        Combines macro context, micro keypoint signal, and follow-through signal
        to produce a final trading recommendation.
        """

        # Extract macro context
        macro_dir = context.get("price_slope",
                                "flat")  # e.g., 'strong up', 'mild up', 'flat', 'mild down', 'strong down'
        macro_volume = context.get("volume_slope", "flat")
        macro_rsi = context.get("rsi_trend", "neutral->neutral")  # e.g., 'neutral->overbought'
        macro_macd = context.get("macd_trend", "neutral->neutral")  # e.g., 'bullish->bearish'
        macro_vwap = context.get("vwap_stance", "mixed")  # e.g., 'mostly above', 'mostly below', 'mixed'
        macro_signal = context.get("context_signal", "range-bound")  # e.g., 'strong uptrend overall'

        # Micro signals
        # Example keypoint signals: 'momentum peak', 'buyer exhaustion', 'calm peak', etc.
        # Example follow-through signals: 'confirmed up', 'confirmed down', 'no follow-through (mixed)'
        micro_kp = kp_signal
        micro_ft = ft_signal

        # Initialize a recommendation
        recommendation = "Hold / No Clear Trade"

        # 1) Senior trader might first see if there's a strong macro uptrend or downtrend
        if "uptrend" in macro_signal:
            # Macro is bullish
            if structure == "valley":
                # If we have a valley in a strong uptrend, potential long
                if "momentum valley" in micro_kp or "strong demand absorption" in micro_kp:
                    if "confirmed up" in micro_ft:
                        recommendation = "Go Long (Bullish Reversal Confirmed)"
                    else:
                        recommendation = "Watch for Bullish Confirmation (Uptrend + Potential Valley)"
                elif "false breakdown" in micro_kp:
                    if "confirmed up" in micro_ft:
                        recommendation = "Bear Trap -> Go Long"
                    else:
                        recommendation = "Possible Bear Trap, Wait for More Confirmation"
                else:
                    # Calm or soft valley
                    recommendation = "Potential Buy on Dip (Macro Uptrend) but Weak Local Signal"
            else:
                # structure == 'peak'
                # In a strong uptrend, a peak might just be a local pullback
                if "momentum peak" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Short-Term Pullback, but Long-Term Uptrend"
                    else:
                        recommendation = "Uptrend Momentum Peak, Consider Partial Profit"
                elif "buyer exhaustion" in micro_kp:
                    recommendation = "Possible Reversal, Watch for Bearish Follow-Through"
                else:
                    recommendation = "Peak in Uptrend - Potential Minor Correction"

        elif "downtrend" in macro_signal:
            # Macro is bearish
            if structure == "peak":
                # Peak in a strong downtrend is a potential short entry
                if "momentum peak" in micro_kp or "buyer exhaustion" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Go Short (Bearish Continuation Confirmed)"
                    else:
                        recommendation = "Watch for Bearish Confirmation (Downtrend + Peak)"
                elif "volume spike reversal" in micro_kp:
                    if "confirmed down" in micro_ft:
                        recommendation = "Volume Spike -> Short Entry Confirmed"
                    else:
                        recommendation = "Possible Reversal Spike, Wait for More Confirmation"
                else:
                    recommendation = "Peak in Downtrend - Potential Sell Rally"
            else:
                # structure == 'valley'
                # In a strong downtrend, a valley might be a local bounce
                if "momentum valley" in micro_kp:
                    recommendation = "Potential Bearish Retracement, Wait for Confirmation"
                elif "false breakdown" in micro_kp:
                    if "confirmed up" in micro_ft:
                        recommendation = "Bullish Divergence, Possible Short Squeeze"
                    else:
                        recommendation = "False Breakdown, but No Confirmation Yet"
                else:
                    recommendation = "Weak Valley in Downtrend, Could Break Lower"

        else:
            # Macro is range-bound or mixed
            if "confirmed up" in micro_ft:
                recommendation = "Range-Bound but Micro Up -> Possible Quick Long"
            elif "confirmed down" in micro_ft:
                recommendation = "Range-Bound but Micro Down -> Possible Quick Short"
            else:
                recommendation = "Sideways Market - Scalping or Wait for Clear Trend"

        # 2) Refine recommendation with RSI or MACD extremes
        # e.g., if RSI ended 'overbought' and we have a peak, that might strengthen a short call
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

        # 3) Optionally factor in VWAP stance
        if "above" in macro_vwap:
            recommendation += " | Price Mostly Above VWAP => Bullish Lean"
        elif "below" in macro_vwap:
            recommendation += " | Price Mostly Below VWAP => Bearish Lean"

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

        You can expand the logic as needed.
        """
        rec_lower = recommendation.lower()

        # 1) Define a base "keyword -> base score" map
        # If multiple strong keywords appear, we can pick the largest absolute or sum them, etc.
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

        # 2) Define multipliers for phrases like "possible", "watch", "caution"
        # If these appear, we'll multiply the final absolute score by the multiplier.
        multiplier_map = {
            "possible": 0.5,
            "watch": 0.7,
            "caution": 0.7,
            "weak": 0.6
        }

        # 3) Find the strongest base keyword
        base_score = 0.0
        for phrase, val in base_map.items():
            if phrase in rec_lower:
                # Option A: pick the phrase with the largest absolute score
                if abs(val) > abs(base_score):
                    base_score = val

        # 4) Apply multipliers if present
        # e.g., if "possible" in text => multiply absolute value by 0.5
        final_score = base_score
        for phrase, mult in multiplier_map.items():
            if phrase in rec_lower:
                final_score = final_score * mult

        # 5) If no strong keyword found, we default to 0 => hold
        # or if the recommendation was "Hold / No Clear Trade"
        if final_score == 0.0 and "hold" not in rec_lower:
            # There's a chance the text is truly neutral
            # You can decide if you want to keep it at 0 or do something else
            final_score = 0.0

        # Ensure final_score is clipped in [-1, 1]
        final_score = max(min(final_score, 1.0), -1.0)

        return final_score

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

            if len(peaks) and len(new_peaks) and index > peaks[-1] + 1:
                print(f"{limiter} peaks found {new_peaks} @{index}")
                for p in new_peaks:
                    prev_peak = max((peak for peak in peaks if peak < p), default=None)
                    if prev_peak is not None:
                        result = self.analyze_pivot(prev_peak, p, structure='peak')
                        print(f"🔴 {result}")
                        if result['trading_decision']['score'] < -0.6:
                            position = -1
                prev_peaks.update(peaks)

            if len(valleys) and index > valleys[-1] + 1 and len(new_valleys):
                print(f"{limiter} valleys found {new_valleys} @{index}")
                for v in new_valleys:
                    if v in vol_peaks or v in vol_valleys:
                        prev_valley = max((valley for valley in valleys if valley < v), default=None)
                        if prev_valley is not None:
                            result = self.analyze_pivot(prev_valley, v, structure='valley')
                            print(f"🔴 {result}")
                            if result['trading_decision']['score'] > 0.6:
                                position = 1
                prev_valleys.update(valleys)

            positions.append(position)
        data['position'] = positions
        self.data = data
        # self.snapshot([20, 100])

    def signal(self):
        self.candle()

