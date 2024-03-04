from strategy import Strategy
from collections import deque


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
        prev_signals = deque(maxlen=5)  # Keep track of the last 5 signals
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
            """
            1) start to accelerate upward in a rising trend - BUY
            2) start to decelerate downward in a dropping trend - BUY
            3) start to accelerate downward in a dropping trend - SELL
            4) start to decelerate upward in a rising trend - SELL
            """

            strength_momentum, decelerate_drop, convergency = False, False, False
            dropping = row['macd_derivative'] < 0 and row['signal_line_derivative'] < 0 or row['macd_derivative'] < row['signal_line_derivative']
            rising = row['macd_derivative'] > row['signal_line_derivative'] > 0
            if len(prev_signals) > 1:
                strength_momentum = row['strength_2nd_derivative'] > 0 and row['strength_2nd_derivative'] > prev_signals[-1][2]
                decelerate_drop = prev_signals[-1][0] < row['macd_derivative'] < row['signal_line_derivative'] < prev_signals[-1][1]
                converging = abs(row['macd_derivative'] - row['signal_line_derivative']) < abs(prev_signals[-1][0] - prev_signals[-1][1]) or (row['macd_derivative'] - row['signal_line_derivative']) * (prev_signals[-1][0] - prev_signals[-1][1]) < 0

            # 2) start to decelerate downward in a dropping trend - BUY
            if strength_momentum and dropping and converging:
                position = 1

            if strength_momentum and rising and not converging:
                position = 1
            if len(prev_signals) > 1 and row['signal_line_derivative'] > row['macd_derivative'] > 0 and prev_signals[-1][0] > prev_signals[-1][1]:
                position = -1

            positions.append(position)
            prev_signals.append((row['macd_derivative'], row['signal_line_derivative'], row['strength_2nd_derivative']))

        data['position'] = positions

    def signal(self):
        self.macd_derivatives()
