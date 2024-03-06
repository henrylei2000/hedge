from strategy import Strategy
from collections import deque


def detect_significance(historical_data, new_point, boundary_ratio=0.25):
    # Calculate the minimum and maximum values within the recent window
    if not len(historical_data):
        return False, False, False, False
    min_value = min(historical_data)
    max_value = max(historical_data)

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


class MACDStrategy(Strategy):

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
            if len(prev_signals) > 1 and row['macd_derivative'] > 0 > row['macd'] > row['signal_line'] and prev_signals[-1][0] < prev_signals[-1][1]:
                position = 1

            if len(prev_signals) > 1 and row['macd_derivative'] < 0 < row['macd'] < row['signal_line'] and prev_signals[-1][0] > prev_signals[-1][1]:
                position = -1

            positions.append(position)
            prev_signals.append((row['macd'], row['signal_line']))

        data['position'] = positions

    def macd_derivatives(self):
        self.macd_simple()
        data = self.data
        prev_macd_derivatives = deque(maxlen=30)  # Keep track of the last 30 signals
        prev_signal_line_derivatives = deque(maxlen=30)
        prev_strength_2nd_derivative = deque(maxlen=30)
        prev_macd_strength = deque(maxlen=30)
        wait = 3
        positions = []  # Store updated signals

        # Calculate the first derivative of MACD
        data['macd_derivative'] = data['macd'].diff()
        data['signal_line_derivative'] = data['signal_line'].diff()
        data['strength_2nd_derivative'] = data['macd_strength'].diff().diff()

        # Initialize Signal column with zeros
        data['position'] = 0

        data.dropna(subset=['macd_derivative'], inplace=True)

        for index, row in data.iterrows():
            position = 0

            strength, macd_derivative, signal_line_derivative = row['macd'] - row['signal_line'], row['macd_derivative'], row['signal_line_derivative']
            strength_2nd_derivative = row['strength_2nd_derivative']

            if len(prev_macd_derivatives) >= wait:
                significance = detect_significance(prev_macd_strength, row['macd_strength'], 0.1)

                if significance[1]:
                    print(f"{row['close']:.3f} @{index} {significance}")
                    if prev_macd_derivatives[-1] > macd_derivative > 0:
                        if prev_signal_line_derivatives[-1] > signal_line_derivative > 0:
                            if strength_2nd_derivative < prev_strength_2nd_derivative[-1] and strength_2nd_derivative < 0:
                                position = -1

                if significance[0]:
                    if prev_macd_derivatives[-1] < macd_derivative and prev_macd_derivatives[-1] < 0:
                        if prev_signal_line_derivatives[-1] < 0 and signal_line_derivative < 0:
                            if prev_strength_2nd_derivative[-1] < strength_2nd_derivative and strength_2nd_derivative > 0:
                                position = 1

            positions.append(position)
            prev_macd_derivatives.append(row['macd_derivative'])
            prev_signal_line_derivatives.append(row['signal_line_derivative'])
            prev_strength_2nd_derivative.append(row['strength_2nd_derivative'])
            prev_macd_strength.append(row['macd_strength'])

        data['position'] = positions

        data.to_csv(f"{self.symbol}.csv")

    def signal(self):
        self.macd_derivatives()
