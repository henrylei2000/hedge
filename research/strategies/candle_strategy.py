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
        # **Key-Point Analysis (Peak, Valley, or Breakout)**
        key_point_signal = "neutral"
        key_vol = data['volume'].iloc[p - 1:p + 2]
        key_price = data['close'].iloc[p - 1:p + 2]
        key_vwap = data['vwap'].iloc[p - 1:p + 2]

        key_vol_trend = np.polyfit(range(3), key_vol, 1)[0]  # Volume trend at key point
        key_price_trend = np.polyfit(range(3), key_price, 1)[0]  # Price trend at key point
        key_vwap_trend = np.polyfit(range(3), key_vwap, 1)[0]  # VWAP trend at key point

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

        print(f"\t\t\t\t🔴[{p} - {structure}] Signal: {key_point_signal}")
        return key_point_signal

    def summarize(self, p, index, peaks, valleys):
        patterns, trading_signals = [], []

        data = self.data
        ca = self.CandleAnalyzer(self)
        ca.atr()
        ca.rvol()
        ca.cluster_volume()

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
        follow_through = ca.follow_through(p + 1, end + 1)
        phases = [(start, p), (p + 1, end + 1)]

        for phase in phases:
            start, end = phase[0], phase[1]
            if end - start > 1:
                cv, c, trend_v = ca.cluster(start, end)
                print(f"\t\t\t\t🔴[{start} - {end - 1}] vol {cv:3d}, trend_v {trend_v:3d}, candle {c} at {structure} @{p}")

                vol = data['volume'].iloc[start:end]
                nvol = data['normalized_volume'].iloc[start:end]
                vwap = data['vwap'].iloc[start:end]
                price = data['close'].iloc[start:end]

                vol_average = vol.mean()
                avg_increase = (vol.iloc[-1] - vol.iloc[0]) / (end - start)
                vol_std = vol.std()
                vol_max_min_ratio = vol.max() / max(vol.min(), 1)  # Avoid zero division
                vol_trend_slope = np.polyfit(range(end - start), vol, 1)[0]  # Linear regression slope
                price_trend_slope = np.polyfit(range(end - start), price, 1)[0]  # Price trend slope
                vwap_trend_slope = np.polyfit(range(end - start), vwap, 1)[0]  # VWAP trend slope
                price_change = price.iloc[-1] - price.iloc[0]  # Absolute price movement

                if vol_trend_slope > 0:
                    if vol_std / vol_average < 0.2:
                        volume_pattern = "gradual increase"
                    elif vol_max_min_ratio > 2:
                        volume_pattern = "strong increase"
                    else:
                        volume_pattern = "erratic increase"
                else:
                    if vol_std / vol_average < 0.2:
                        volume_pattern = "gradual decrease"
                    elif vol_max_min_ratio > 2:
                        volume_pattern = "strong decrease"
                    else:
                        volume_pattern = "erratic decrease"

                patterns.append(volume_pattern)

                # **Determine General Trading Scenario**
                if price_trend_slope > 0 and volume_pattern == "gradual increase":
                    trading_signal = "bullish continuation"
                elif price_trend_slope < 0 and volume_pattern == "gradual decrease":
                    trading_signal = "bearish continuation"
                elif price_change < 0 and volume_pattern == "strong increase":
                    trading_signal = "bear trap (liquidity grab) - reversal up"
                elif price_change > 0 and volume_pattern == "strong decrease":
                    trading_signal = "bull trap (liquidity grab) - reversal down"
                else:
                    trading_signal = "neutral"
                trading_signals.append(trading_signal)

                print(f"\t\t\t\tVolume Pattern: {volume_pattern}, Trading Signal: {trading_signal}")
                print(
                    f"\t\t\t\tAvg Increase: {avg_increase:.2f}, Vol Std: {vol_std:.2f} / {vol_average:.2f}, Vol Max/Min Ratio: {vol_max_min_ratio:.2f}, Vol Trend Slope: {vol_trend_slope:.2f}")
                print(f"\t\t\t\tPrice Trend Slope: {price_trend_slope:.2f}, VWAP Trend Slope: {vwap_trend_slope:.2f}")
            else:
                print(f"\t\t\t\t🔴[{start} - {end - 1}] Too close to call, wait for {2 - end + start} more bar(s) at {start + 1}")

        return patterns[0] + ', ' + trading_signals[0], key_point_signal, follow_through

    def candle(self):
        self.normalized('trending')
        self.normalized('volume')
        self.normalized('vwap')
        ca = self.CandleAnalyzer(self)
        data = ca.analyze()
        distance = 5
        prev_peaks, prev_valleys = set(), set()
        prev_vol_peaks, prev_vol_valleys = set(), set()
        positions = []
        base = data.iloc[0]['close']
        for index in range(len(data)):
            row = data.iloc[index]
            position = 0
            visible_rows = data.loc[:index]
            prices, highs, lows, volumes = visible_rows['close'], visible_rows['high'], visible_rows['low'], visible_rows['volume']

            print(f"{index:3d} 📈{row['macd'] * 100:4.1f}", end=" ")
            print(f"🚿{row['normalized_volume']:3d}, 🏹{int(row['tension']):4d}", end=" ")
            candle = f"🕯️{int(row['span'] / base * 10000)} ({row['upper_wick'] * 100:.0f} {row['body_size'] * 100:.0f} {row['lower_wick'] * 100:.0f})"
            print(f"{candle:18} {row['candlestick']}")

            peaks, _ = find_peaks(prices, distance=distance)
            valleys, _ = find_peaks(-prices, distance=distance)
            new_peaks = [p for p in peaks if p > distance and p not in prev_peaks and index - p < 5]
            new_valleys = [v for v in valleys if v > distance and v not in prev_valleys and index - v < 5]

            vol_peaks, _ = find_peaks(volumes, distance=distance)
            vol_valleys, _ = find_peaks(-volumes, distance=distance)
            new_vol_peaks = [p for p in vol_peaks if p > distance and p not in prev_vol_peaks and index - p < 5]
            new_vol_valleys = [v for v in vol_valleys if v > distance and v not in prev_vol_valleys and index - v < 5]

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
        # self.snapshot([350, 389])

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

        def get_cluster_info(self, start, end, label):
            vol, candle, trend_vol = self.cluster(start, end)
            return f"{label} vol {vol:3d}, trend_v {trend_vol:3d}, candle {candle}"

        def cluster(self, start, end):
            data = self.data
            data['normalized_volume_diff'] = data['normalized_volume'].diff().fillna(0).astype(int)
            cluster_volume, cluster_volume_diff = 0, 0
            cluster_upper, cluster_body, cluster_lower = 0, 0, 0
            for i in range(start, end):
                row = data.iloc[i]
                upper = int(row['upper_wick'] * 100)
                body = int(row['body_size'] * 100)
                lower = int(row['lower_wick'] * 100)
                volume = row['normalized_volume']
                volume_diff = row['normalized_volume_diff']

                cluster_volume += volume
                cluster_upper += upper
                cluster_lower += lower
                cluster_body += body
                cluster_volume_diff += volume_diff

            return cluster_volume // (end - start), [cluster_upper, cluster_body, cluster_lower], cluster_volume_diff // (end - start)

        def follow_through(self, start, end):
            data = self.data
            signal = 'neutral'
            for i in range(start, end):
                row = data.iloc[i]
                body = int(row['body_size'] * 100)
                if body > 40:
                    return 'reversal up'
                if body < -30:
                    return 'reversal down'
            return signal

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

            base = data.iloc[0]['close']

            for idx, row in data.iterrows():
                upper = int(row['upper_wick'] * 100)
                body = int(row['body_size'] * 100)
                lower = int(row['lower_wick'] * 100)
                span = int(row['span'] / base * 10000)
                volume = row['normalized_volume']
                trend = row['normalized_trending']
                trend_clustered_volume_avg = row['trend_clustered_volume_avg']
                normalized_vwap = row['normalized_vwap']
                tension = row['tension']
                price = row['close']
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

                if abs(body) > 80 and span > 25:
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

                if abs(tension) > 25:
                    signal.append("Reversal: High tension between price and VWAP, potential reversion move")

                # Resistance (long upper wick with high volume strengthens the signal)
                if strength > 0 and upper > 30 and span > 15:
                    if volume > 60:
                        signal.append("Potential strong resistance (long upper wick, high volume)")
                        todos.append((idx, 'strong resistance'))
                    elif volume > 40:
                        signal.append("Potential weak resistance (long upper wick, moderate volume)")
                        todos.append((idx, 'weak resistance'))

                # Support (long lower wick with high volume confirms demand)
                if strength < 0 and lower > 25 and span > 15:
                    if volume > 60:
                        signal.append("Potential strong support (long lower wick, high volume)")
                    elif volume > 40:
                        signal.append("Potential weak support (long lower wick, moderate volume)")

                # Buying Pressure (strong bullish body, rising volume, and trend shift)
                if body > 50 and volume > 60 and trend > 10:
                    signal.append("Bullish: Buying pressure detected (body, vol, trend)")

                # Selling Pressure (strong bearish body, rising volume, and downward trend shift)
                if body < -50 and volume > 60 and trend < -10:
                    signal.append("Bearish: Selling pressure detected (body, vol, trend)")

                prev_row = data.iloc[idx - 1]
                prev_body = int(prev_row['body_size'] * 100)
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
                if 0 < trend < 10 and abs(tension) > 35 and volume < 40:
                    signal.append("Strong VWAP mean reversion detected, trend weakening, low volume")

                if 0 < trend < 10 and rvol < self.rvol_threshold and volume < 40:
                    signal.append("Weak momentum (RVOL low), trend losing strength")

                signals.append((idx, ", ".join(signal) if signal else ""))
                positions.append(position)

            self.data['candlestick'] = self.data.index.map(dict(signals))
            self.data['position'] = positions

            return self.data
