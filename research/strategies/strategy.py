import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import deque


class Strategy:
    def __init__(self, symbol='TQQQ', start='2025-01-28 09:30', end='2024-01-28 16:00', api=None, context=None):
        self.symbol = symbol
        self.start = pd.Timestamp(start, tz='America/New_York').tz_convert('UTC')
        self.end = pd.Timestamp(end, tz='America/New_York').tz_convert('UTC')
        self.api = api
        self.context = context
        self.data = None
        self.data5 = None
        self.pnl = 0.00
        self.trades = 0
        self.init_balance = 10000
        self.num_buckets = 4

    def backtest(self):
        if self.api:
            self.retrospective()
            self.download()
        else:
            self.data = pd.read_csv('TQQQ.csv')
        if not self.data.empty:
            self.prepare()
            self.sanitize()
            self.signal()
            self.bucket_trade()
            # self.plot()
        else:
            print("No data found, please verify symbol and date range.")

    def retrospective(self):
        prices = self.context['close']
        peaks, _ = find_peaks(prices, distance=5)
        valleys, _ = find_peaks(-prices, distance=5)
        peak_indices = np.array(peaks)
        peak_prices = prices.iloc[peaks]
        a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)

        valley_indices = np.array(valleys)
        valley_prices = prices.iloc[valleys]
        a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)

        next_day_index = len(prices)
        predicted_peak = a_peaks * next_day_index + b_peaks
        predicted_valley = a_valleys * next_day_index + b_valleys

        last_peak_index = peak_indices[-1] if len(peak_indices) > 0 else -1
        last_valley_index = valley_indices[-1] if len(valley_indices) > 0 else -1

        if last_peak_index > last_valley_index:
            last_abnormal_day = 'Peak'
        else:
            last_abnormal_day = 'Valley'

        if last_abnormal_day == 'Peak':
            next_day_prediction = 'Valley'
        else:
            next_day_prediction = 'Peak'

        prediction = {
            'Next Day Index': next_day_index,
            'Predicted Peak': predicted_peak,
            'Predicted Valley': predicted_valley,
            'Prediction Type': next_day_prediction
        }

        return prediction

    def download(self):
        data = self.api.get_bars(self.symbol, '1T', start=self.start.isoformat(), end=self.end.isoformat()).df
        if not data.empty:
            data.index = data.index.tz_convert('US/Eastern')  # convert timestamp index to Eastern Timezone (EST)
            data = data.between_time('9:30', '15:59')  # filter rows between 9:30am and 4:00pm EST
            data = data.reset_index()  # converts timestamp index into a column
            data.rename(columns={'index': 'timestamp'}, inplace=True)
        self.data = data

    def prepare(self):
        short_window, long_window, signal_window = 12, 26, 9  # 9, 21, 6
        data = self.data
        data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
        data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()
        data['trending'] = data['close'] - data['close'].ewm(alpha=0.3, adjust=False).mean()
        data['macd'] = data['short_ma'] - data['long_ma']
        data['macd_signal'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
        data['strength'] = data['macd'] - data['macd_signal']
        data['rsi'] = self.calculate_rsi(data['close'], signal_window)

        price_change_ratio = data['close'].pct_change()
        data['vpt'] = (price_change_ratio * data['volume']).cumsum()
        data['rolling_vpt'] = data['vpt'].rolling(window=signal_window).mean()
        data['obv'] = (data['volume'] * ((data['close'] - data['close'].shift(1)) > 0).astype(int) -
                       data['volume'] * ((data['close'] - data['close'].shift(1)) < 0).astype(int)).cumsum()
        data['rolling_obv'] = data['obv'].rolling(window=signal_window).mean()
        data['volume_sma'] = data['volume'].rolling(window=signal_window, min_periods=1).mean()
        data['rvol'] = data['volume'] / data['volume_sma']
        data['a/d'] = Strategy.ad_line(data['close'], data['high'], data['low'], data['volume'])

        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        data['tpv'] = data['typical_price'] * data['volume']
        data['tension'] = ((data['close'] - data['vwap']) / data['close'] * 10000).rolling(window=1).mean().fillna(0)

        high_low = data['high'] - data['low']
        data['span'] = high_low
        data['body'] = (data['close'] - data['open']) / data['span']
        data['upper'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['span']
        data['lower'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['span']

        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = true_range.rolling(window=signal_window).mean()

        group_size = 5
        data['group'] = data.index // group_size
        grouped = data.groupby('group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'tpv': 'sum'
        })
        grouped['vwap'] = grouped['tpv'] / grouped['volume']
        grouped['rsi'] = self.calculate_rsi(grouped['close'], period=3)
        self.data5 = grouped

        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

    def sanitize(self):
        pass

    def signal(self):
        pass

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def classify_rsi(rsi_value):
        """
        Simple RSI classification:
          - > 70: overbought
          - < 30: oversold
          - Otherwise: neutral
        """
        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        else:
            return "neutral"

    @staticmethod
    def classify_macd(macd_value, macd_signal_value):
        """
        Simple MACD classification:
          - macd_value > macd_signal_value: bullish
          - macd_value < macd_signal_value: bearish
          - Otherwise: neutral
        """
        if macd_value > macd_signal_value:
            return "bullish"
        elif macd_value < macd_signal_value:
            return "bearish"
        else:
            return "neutral"

    @staticmethod
    def slope_classification(slope, mild_thresh=0.01, steep_thresh=0.05):
        """
        Classify slope into:
          - strong up
          - mild up
          - flat
          - mild down
          - strong down
        """
        if slope > steep_thresh:
            return "strong up"
        elif slope > mild_thresh:
            return "mild up"
        elif slope < -steep_thresh:
            return "strong down"
        elif slope < -mild_thresh:
            return "mild down"
        else:
            return "flat"

    @staticmethod
    def rearrange_valley_peak(valley_indices, valley_prices, peak_indices, peak_prices, alternative_valley):
        valley_indices = np.array(valley_indices)
        valley_prices = np.array(valley_prices)
        peak_indices = np.array(peak_indices)
        peak_prices = np.array(peak_prices)

        # Handle scenario 1: if the first peak appears before the first valley
        if len(peak_indices) and len(valley_indices) and peak_indices[0] < valley_indices[0]:
            valley_indices = np.insert(valley_indices, 0, 0)
            valley_prices = np.insert(valley_prices, 0, alternative_valley)

        # Create lists to store the corrected indices, peak indices, and valley indices
        corrected_indices = []
        corrected_peak_indices = []
        corrected_valley_indices = []

        # Start with the first valley
        i = j = 0
        while i < len(valley_indices) and j < len(peak_indices):
            # Handle consecutive valleys using argmin
            start_i = i
            while i + 1 < len(valley_indices) and valley_indices[i + 1] < peak_indices[j]:
                i += 1
            # Find the valley with the minimum price in the range
            min_price_idx = start_i + np.argmin(valley_prices[start_i:i + 1])
            corrected_indices.append(valley_indices[min_price_idx])
            corrected_valley_indices.append(valley_indices[min_price_idx])
            i += 1

            # Handle consecutive peaks using argmax
            start_j = j
            while j + 1 < len(peak_indices) and i < len(valley_indices) and peak_indices[j + 1] < valley_indices[i]:
                j += 1
            # Find the peak with the maximum price in the range
            max_price_idx = start_j + np.argmax(peak_prices[start_j:j + 1])
            corrected_indices.append(peak_indices[max_price_idx])
            corrected_peak_indices.append(peak_indices[max_price_idx])
            j += 1

        # Handle the case where only valleys are left
        if i < len(valley_indices):
            # Find the index with the minimum price among the remaining valleys
            min_price_idx = i + np.argmin(valley_prices[i:])
            corrected_indices.append(valley_indices[min_price_idx])
            corrected_valley_indices.append(valley_indices[min_price_idx])

        # Handle the case where only peaks are left
        if j < len(peak_indices):
            # Find the index with the maximum price among the remaining peaks
            max_price_idx = j + np.argmax(peak_prices[j:])
            corrected_indices.append(peak_indices[max_price_idx])
            corrected_peak_indices.append(peak_indices[max_price_idx])

        return np.array(corrected_indices), np.array(corrected_valley_indices), np.array(corrected_peak_indices)

    @staticmethod
    def ad_line(prices, high, low, volume):
        money_flow_multiplier = ((prices - low) - (high - prices)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line

    def trade(self):
        balance = self.init_balance
        shares_held = 0
        trades = 0
        if 'position' not in self.data.columns:
            self.data['position'] = self.data['signal']
        for index, row in self.data.iterrows():
            if shares_held > 0 > row['position']:
                trades += 1
                balance += row['close'] * shares_held
                print(f"Sold at: ${row['close']:.2f} x {shares_held}  @{index}")
                print(f"Trade {trades} ------------- Balance: ${balance:.2f} [macd {row['macd'] * 100:.3f}]")
                shares_held = 0

            elif balance > 0 and row['position'] > 0:
                shares_bought = balance / row['close']
                balance -= row['close'] * shares_bought
                shares_held += shares_bought
                if shares_bought:
                    print(f"share bought: {shares_bought:.2f}")
                    print(f"Bought at: ${row['close']:.2f} x {shares_bought}  @{index} [macd {row['macd'] * 100:.3f}]")

        final_balance = balance + (shares_held * self.data['close'].iloc[-1])
        self.pnl = final_balance - self.init_balance
        print(f"Initial Balance: ${self.init_balance:.2f} -------- Final Balance: ${final_balance:.2f} "
              f"\n-------{self.symbol}---------- PnL: ${self.pnl:.2f}")

    def bucket_trade(self):
        initial_balance = self.init_balance
        num_buckets = self.num_buckets  # 4
        initial_bucket_value = initial_balance / num_buckets
        buckets = [{'in_use': False, 'shares': 0, 'buy_price': 0, 'bucket_value': initial_bucket_value} for _ in
                   range(num_buckets)]
        total_balance = initial_balance  # Updated to track total balance across all buckets
        if 'position' not in self.data.columns:
            self.data['position'] = self.data['signal']
        stock_data = self.data
        price = 0
        for i, row in stock_data.iterrows():
            price = row['close']  # Assuming the 'close' column has the stock price
            position = row['position']

            if position > 0:  # Buy signal
                num_buckets_to_buy = int(position)
                available_buckets = [bucket for bucket in buckets if not bucket['in_use']]
                buckets_to_buy = min(num_buckets_to_buy, len(available_buckets))
                for bucket in available_buckets[:buckets_to_buy]:
                    bucket['in_use'] = True
                    bucket['buy_price'] = price
                    bucket['shares'] = bucket['bucket_value'] / price
                    self.trades += 1

            elif position < 0:  # Sell signal
                num_buckets_to_sell = -int(position)
                in_use_buckets = [bucket for bucket in buckets if bucket['in_use']]
                buckets_to_sell = min(num_buckets_to_sell, len(in_use_buckets))  # Ensure num_buckets_to_sell is valid

                for bucket in in_use_buckets[:buckets_to_sell]:
                    sell_value = bucket['shares'] * price
                    pnl = sell_value - bucket['bucket_value']
                    total_balance += pnl  # Update total balance
                    bucket['in_use'] = False
                    bucket['shares'] = 0
                    bucket['buy_price'] = 0
                    bucket['bucket_value'] = sell_value  # Update bucket value with the result of the trade

        final_balance = sum(bucket['bucket_value'] for bucket in buckets if not bucket['in_use']) + \
            sum(bucket['shares'] * price for bucket in buckets if bucket['in_use'])
        total_pnl = final_balance - initial_balance
        self.pnl = total_pnl
        return total_pnl

    def normalized(self, column='volume', window_size=20):
        data = self.data
        normalized_columns = []
        rolling_window = deque(maxlen=window_size)  # Keep a limited history
        values = data[column]
        for value in values:
            rolling_window.append(value)  # Add new value, automatically removes oldest if full
            band = max(abs(x) for x in rolling_window)  # Get max in the window
            normalized_value = int((value / band) * 100) if band else 0
            normalized_columns.append(normalized_value)
        data['normalized_' + column] = normalized_columns

    def snapshot(self, span, indicators=None):
        if indicators is None:
            indicators = ['tension', 'trending']
        if span[1] == -1 or span[1] > 389:
            span[1] = 389
        if span[1] - span[0] < 10:
            return

        rows = self.data.iloc[span[0]:span[1]]
        prices, lows, highs = rows['close'], rows['low'], rows['high']

        distance = 5
        peaks, _ = find_peaks(prices, distance=distance)
        peak_indices = np.array(peaks)
        valleys, _ = find_peaks(-prices, distance=distance)
        valley_indices = np.array(valleys)
        low_valleys, _ = find_peaks(-lows, distance=distance)
        low_valley_indices = np.array(low_valleys)
        high_peaks, _ = find_peaks(highs, distance=distance)
        high_peak_indices = np.array(high_peaks)

        buy_signals = rows[rows['position'] > 0]
        sell_signals = rows[rows['position'] < 0]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                            gridspec_kw={'height_ratios': [3, 2, 2]})

        ax1.plot(prices, label='Price', color='blue')
        ax1.set_title(f"{self.symbol}, {self.start.strftime('%Y-%m-%d')} {span}")
        ax1.set_ylabel('Price')
        ax1.legend()
        filtered_low_valley_indices = low_valley_indices[~np.isin(low_valley_indices, valley_indices)]
        ax1.plot(lows.iloc[filtered_low_valley_indices], 'go', label='low valleys')
        filtered_high_peak_indices = high_peak_indices[~np.isin(high_peak_indices, peak_indices)]
        ax1.plot(highs.iloc[filtered_high_peak_indices], 'ro', label='high peaks')
        for peak in peak_indices:
            ax1.annotate(f'{span[0] + peak}',
                         (prices.index[peak], prices.iloc[peak]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, 10),  # Offset text by 10 points above the peak
                         ha='center',  # Center-align the text
                         fontsize=9, color='red')  # You can adjust the font size if needed
        # ax1.plot(prices.iloc[valley_indices], 'go', label='Valleys')
        for valley in valley_indices:
            ax1.annotate(f'{span[0] + valley}',
                         (prices.index[valley], prices.iloc[valley]),
                         textcoords="offset points",  # Positioning relative to the peak
                         xytext=(0, -10),  # Offset text by 10 points below the valley
                         ha='center',  # Center-align the text
                         fontsize=9, color='green')  # You can adjust the font size if needed

        # **Plot candles with wicks (High-Low)**
        ax1.plot(buy_signals.index, buy_signals['close'], 'g^', markersize=12, alpha=1, label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['close'], 'rv', markersize=12, alpha=1, label='Sell Signal')
        ax1.vlines(rows.index, rows['low'], rows['high'], color='black', alpha=.5, linewidth=1)
        colors = rows.apply(lambda row: 'green' if row['close'] > row['open'] else 'red', axis=1)
        ax1.bar(rows.index, abs(rows['close'] - rows['open']), bottom=rows[['open', 'close']].min(axis=1), color=colors,
                alpha=.5, edgecolor='none')

        ax4 = ax1.twinx()
        ax4.bar(rows.index, rows['volume'].values,  color='gray', alpha=0.2, label='volume')

        for i in range(len(indicators)):
            indicator = indicators[i]
            ax = ax2 if i == 0 else ax3
            obvs = rows[indicator]
            obv_peaks, _ = find_peaks(obvs, distance=distance)
            obv_peak_indices = np.array(obv_peaks)
            obv_valleys, _ = find_peaks(-obvs, distance=distance)
            obv_valley_indices = np.array(obv_valleys)
            ax.plot(rows[indicator], label=f"{indicator}", color='lightblue')
            if indicator not in ['rsi', 'volume']:
                ax.axhline(y=0, color='r', linestyle='--')
            # ax2.plot(obvs.iloc[obv_peak_indices], 'ro', label='peaks')
            # Annotate each peak with its value
            for peak in obv_peak_indices:
                ax.annotate(f'{span[0] + peak}',
                            (obvs.index[peak], obvs.iloc[peak]),
                            textcoords="offset points",  # Positioning relative to the peak
                            xytext=(0, 10),  # Offset text by 10 points above the peak
                            ha='center',  # Center-align the text
                            fontsize=9, color='red')  # You can adjust the font size if needed
            # ax2.plot(obvs.iloc[obv_valley_indices], 'go', label='valleys')
            for valley in obv_valley_indices:
                ax.annotate(f'{span[0] + valley}',
                            (obvs.index[valley], obvs.iloc[valley]),
                            textcoords="offset points",  # Positioning relative to the peak
                            xytext=(0, 10),  # Offset text by 10 points above the peak
                            ha='center',  # Center-align the text
                            fontsize=9, color='green')  # You can adjust the font size if needed
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot(self):
        r = self.data.to_records()
        fig, ax = plt.subplots(figsize=(18, 8))

        ax.plot(np.arange(len(r)), r.close, linewidth=1.2, label='close', color='blue')
        ax.scatter(np.where(r.position > 0)[0], r.close[r.position > 0], marker='o', color='g', alpha=.5, s=120,
                   label='buy')
        ax.scatter(np.where(r.position < 0)[0], r.close[r.position < 0], marker='o', color='r', alpha=.5, s=120,
                   label='sell')

        ax2 = ax.twinx()
        ax2.bar(np.arange(len(r)), r.volume, color='gray', alpha=0.4, width=0.5, label='volume')

        ax.set_title(f"{self.symbol}, {self.start.strftime('%Y-%m-%d')}")
        ax.legend(loc='upper right')

        plt.show()
