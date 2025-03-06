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

    def keypoint(self, p, structure):
        data = self.data
        key_point_signal = "neutral"
        key_vol = data['volume'].iloc[p - 1:p + 2]
        key_price = data['close'].iloc[p - 1:p + 2]

        key_vol_trend = np.polyfit(range(3), key_vol, 1)[0]  # Volume trend at key point
        key_price_trend = np.polyfit(range(3), key_price, 1)[0]  # Price trend at key point

        if structure == "peak":
            if key_vol_trend > 0 and key_price_trend > 0:
                key_point_signal = "momentum peak"
            elif key_vol_trend < 0 < key_price_trend:
                key_point_signal = "buyer exhaustion"
            elif key_vol_trend > 0 > key_price_trend:
                key_point_signal = "volume spike reversal"
            else:
                key_point_signal = "calm peak"
        elif structure == "valley":
            if key_vol_trend > 0 and key_price_trend > 0:
                key_point_signal = "momentum valley"
            elif key_vol_trend < 0 < key_price_trend:
                key_point_signal = "strong demand absorption"
            elif key_vol_trend > 0 > key_price_trend:
                key_point_signal = "false breakdown"
            else:
                key_point_signal = "calm valley"

        return key_point_signal

    def follow_through(self, start, end):
        data = self.data
        signal = 'neutral'
        for i in range(start, end):
            row = data.iloc[i]
            body = int(row['body'] * 100)
            if body > 40:
                return 'reversal up'
            if body < -30:
                return 'reversal down'
        return signal

    def cluster(self, start, end):
        data = self.data
        volume = data.iloc[start:end]['volume'].sum()
        span = data.iloc[start:end]['high'].max() - data.iloc[start:end]['low'].min()
        body = (data.iloc[end - 1]['close'] - data.iloc[start]['open']) / span * 100
        upper = (data.iloc[start:end]['high'].max() - max(data.iloc[end - 1]['close'], data.iloc[start]['open'])) / span * 100
        lower = (min(data.iloc[end - 1]['close'], data.iloc[start]['open']) - data.iloc[start:end]['low'].min()) / span * 100
        return volume // 10000, int(span / data.iloc[start]['close'] * 10000), [int(upper.round()), int(body.round()), int(lower.round())]

    def comment(self, idx):
        data = self.data
        row = data.iloc[idx]
        price = row['close']
        upper = int(row['upper'] * 100)
        body = int(row['body'] * 100)
        lower = int(row['lower'] * 100)
        span = int(row['span'] / price * 10000)
        volume = row['volume'] // 10000
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
            if volume > 15:
                comment.append("Potential strong resistance (long upper wick, high volume)")
            elif volume > 9:
                comment.append("Potential weak resistance (long upper wick, moderate volume)")

        # Support (long lower wick with high volume confirms demand)
        if strength < 0 and lower > 25 and span > 15:
            if volume > 15:
                comment.append("Potential strong support (long lower wick, high volume)")
            elif volume > 9:
                comment.append("Potential weak support (long lower wick, moderate volume)")

        # Buying Pressure (strong bullish body, rising volume, and trend shift)
        if body > 50 and volume > 15 and macd > 0:
            comment.append("Bullish: Buying pressure detected (body, vol, trend)")

        # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
        if body < -50 and volume > 15 and macd < 0:
            comment.append("Bearish: Selling pressure detected (body, vol, trend)")

        # Exit if VWAP reversion is likely
        if 0 < macd and abs(tension) > 35 and volume < 10:
            comment.append("Strong VWAP mean reversion detected, trend weakening, low volume")

        if 0 < macd and rvol < rvol_threshold and volume < 10:
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

    def summarize(self, p, index, peaks, valleys):
        patterns, trading_signals = [], []
        data = self.data
        start, end, structure = 0, -1, ''
        if p in peaks:
            structure = 'peak'
            start = max([x for x in valleys if x < p - 1], default=0)
            end = next((x for x in valleys if x > p + 1), index)
        elif p in valleys:
            structure = 'valley'
            start = max([x for x in peaks if x < p - 1], default=0)
            end = next((x for x in peaks if x > p + 1), index)

        key_point_signal = self.keypoint(p, structure)
        follow_through = self.follow_through(p + 1, end + 1)
        phases = [(start, p), (p, p + 1), (p + 1, end + 1)]

        for phase in phases:
            start, end = phase[0], phase[1]
            cv, span, c = self.cluster(start, end)
            print(f"\t\t\t\t🔴[{start} - {end - 1}] vol {cv:3d}, candle {span} {c} at {structure} @{p}")
            if end - start > 1:
                vol = data['volume'].iloc[start:end]
                vwap = data['vwap'].iloc[start:end]
                price = data['close'].iloc[start:end]

                vol_average = vol.mean()
                vol_std = vol.std()
                vol_max_min_ratio = vol.max() / max(vol.min(), 1)  # Avoid zero division

                duration = end - start  # Length of the period
                vol_trend_slope = np.polyfit(range(duration), vol, 1)[0]
                price_trend_slope = np.polyfit(range(duration), price, 1)[0]
                vwap_trend_slope = np.polyfit(range(duration), vwap, 1)[0]

                price_change = price.iloc[-1] - price.iloc[0]  # Absolute price movement

                # Define slope significance dynamically based on volume magnitude
                steep_slope_threshold = 0.21 * vol_average  # 21% of average volume
                moderate_slope_threshold = 0.08 * vol_average  # 8% of average volume

                # Categorize volume pattern
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

                if price_trend_slope > 0 and volume_pattern == "steady increase":
                    trading_signal = "bullish continuation"
                elif price_trend_slope < 0 and volume_pattern == "steady decrease":
                    trading_signal = "bearish continuation"
                elif price_change < 0 and "strong increase" in volume_pattern:
                    trading_signal = "bear trap (liquidity grab) - reversal up"
                elif price_change > 0 and volume_pattern == "strong decrease":
                    trading_signal = "bull trap (liquidity grab) - reversal down"
                else:
                    trading_signal = "neutral"
                trading_signals.append(trading_signal)

                print(f"\t\t\t\tVolume Pattern: {volume_pattern}, Trading Signal: {trading_signal}")
                print(
                    f"\t\t\t\tVol avg/std: {int(vol_average//10000)}/{int(vol_std//10000)} Trend Slope: {vol_trend_slope / 10000:.1f}, Max/Min: {vol_max_min_ratio:.1f}")
                print(f"\t\t\t\tPrice Trend Slope: {price_trend_slope:.2f}, VWAP Trend Slope: {vwap_trend_slope:.2f}")

        return patterns[0] + ', ' + trading_signals[0], key_point_signal, follow_through

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
        base = data.iloc[0]['close']
        for index in range(len(data)):
            row = data.iloc[index]
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

