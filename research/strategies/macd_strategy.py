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

    def macd_derivatives(self):
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
            if len(prev_signals) > 1 and row['macd_derivative'] + prev_signals[-1] > 0 > row['macd'] and row['signal_line'] < 0:
                position = 1

            if len(prev_signals) > 1 and row['macd_derivative'] + prev_signals[-1] < 0 < row['macd'] and row['signal_line'] > 0:
                position = -1

            positions.append(position)
            prev_signals.append(row['position'])

        data['position'] = positions

    def signal(self):
        self.macd_derivatives()
