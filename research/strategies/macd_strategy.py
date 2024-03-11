from strategy import Strategy
from collections import deque


class MACDStrategy(Strategy):
    def detect_significance(self, index, column, window=26, boundary_ratio=0.25):
        # Calculate the minimum and maximum values within the recent window
        recent_rows = self.data.loc[:index].tail(window+1)[:-1]
        current_row = self.data.loc[index]
        if not len(self.data):
            return False, False, False, False
        min_value = recent_rows[column].min()
        max_value = recent_rows[column].max()

        new_point = current_row[column]
        # Calculate the boundary threshold based on the ratio and range of values
        value_range = max_value - min_value
        boundary_threshold = value_range * boundary_ratio

        # Check if the new point is approaching the minimum or maximum boundary
        approaching_min_boundary = new_point <= (min_value + boundary_threshold)
        approaching_max_boundary = new_point >= (max_value - boundary_threshold)

        # Also check if it's exceeding boundaries
        exceeding_min_boundary = new_point < min_value
        exceeding_max_boundary = new_point > max_value

        return approaching_min_boundary, approaching_max_boundary, exceeding_min_boundary, exceeding_max_boundary

    def macd_simple(self):
        short_window, long_window, signal_window = 9, 26, 6   # 3, 7, 2
        data = self.data
        # Calculate short-term and long-term exponential moving averages
        data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
        data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()

        # Calculate MACD line
        data['macd'] = data['short_ma'] - data['long_ma']
        # Calculate Signal line
        data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
        data['macd_strength'] = data['macd'] - data['signal_line']
        # Generate Buy and Sell signals
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

        # Buy signal: MACD crosses above Signal line
        data.loc[data['macd'] > data['signal_line'], 'signal'] = 1

        # Sell signal: MACD crosses below Signal line
        data.loc[data['macd'] < data['signal_line'], 'signal'] = -1

        data.dropna(subset=['close', 'macd', 'macd_strength', 'signal_line'], inplace=True)

    def macd_cross_line(self):
        self.macd_simple()
        data = self.data
        prev_signals = deque(maxlen=5)  # Keep track of the last 5 signals
        positions = []  # Store updated signals

        # Calculate the first derivative of MACD
        data['macd_derivative'] = data['macd'].diff()

        # Initialize Signal column with zeros
        data['position'] = 0

        data.dropna(subset=['macd_derivative'], inplace=True)

        for index, row in data.iterrows():
            position = 0
            if len(prev_signals) > 1:
                if row['macd_derivative'] > 0 > row['macd'] > row['signal_line'] and prev_signals[-1][0] < prev_signals[-1][1]:
                    position = 1

                if row['macd_derivative'] < 0 < row['macd'] < row['signal_line'] and prev_signals[-1][0] > prev_signals[-1][1]:
                    position = -1

            positions.append(position)
            prev_signals.append((row['macd'], row['signal_line']))

        data['position'] = positions

    def macd_derivatives(self):
        self.macd_simple()
        data = self.data
        prev_macd_derivatives = deque(maxlen=3)  # Keep track of the last 30 signals
        prev_macd_strength = deque(maxlen=3)
        wait = 3
        positions = []  # Store updated signals

        # Calculate the first derivative of MACD
        data['macd_derivative'] = data['macd'].diff()
        data['signal_line_derivative'] = data['signal_line'].diff()

        # Initialize Signal column with zeros
        data['position'] = 0

        data.dropna(subset=['macd_derivative'], inplace=True)

        for index, row in data.iterrows():
            position = 0

            strength, macd_derivative, signal_line_derivative = row['macd'] - row['signal_line'], row['macd_derivative'], row['signal_line_derivative']

            if len(prev_macd_derivatives) >= wait:
                significance = self.detect_significance(index, 'close')
                strength_significance = self.detect_significance(index, 'macd_strength')

                if significance[1]:
                    print(f"PEAK {row['close']:.2f} @{index} {significance} {prev_macd_derivatives[-1]:.3f} > {macd_derivative:.3f} > 0")
                    if prev_macd_derivatives[-1] > macd_derivative > 0:
                        position = -1

                if significance[0]:
                    print(f"VALLEY {row['close']:.2f} @{index} {significance} {prev_macd_derivatives[-1]:.3f} < {macd_derivative:.3f} < 0")
                    if prev_macd_derivatives[-1] < macd_derivative < 0:
                        position = 1

            positions.append(position)
            prev_macd_derivatives.append(row['macd_derivative'])
            prev_macd_strength.append(row['macd_strength'])

        data['position'] = positions

        data.to_csv(f"{self.symbol}.csv")

    def signal(self):
        self.macd_derivatives()
