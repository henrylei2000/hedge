from strategy import Strategy
from collections import deque


class MACDStrategy(Strategy):
    def macd_momentum(self):
        self.macd_simple()
        data = self.data
        prev_signals = deque(maxlen=5)  # Keep track of the last 5 signals
        positions = []  # Store updated signals
        data['macd_momentum'] = data['macd_strength'].diff()
        data.dropna(subset=['macd_momentum'], inplace=True)
        for index, row in data.iterrows():
            position = 0
            signal = row['signal']
            if signal == -1:
                # Sell signal
                if len(prev_signals) > 1 and row['macd_momentum'] < 0 and row['macd_momentum'] < prev_signals[-1]:
                    position = -1 if positions[-1] != 1 else 0
                if len(prev_signals) > 1 and 0 > row['macd_momentum'] > prev_signals[-1] > prev_signals[-2]:
                    position = 1 if positions[-1] != -1 else 0
            elif signal == 1:
                # Buy signal
                if len(prev_signals) > 1 and row['macd_momentum'] > prev_signals[-1] and 0 > prev_signals[-1] > prev_signals[-2] > 0:
                    position = 1 if positions[-1] != -1 else 0
                if len(prev_signals) > 1 and 0 < row['macd_momentum'] < prev_signals[-1] < prev_signals[-2]:
                    position = -1 if positions[-1] != 1 else 0

            positions.append(position)
            prev_signals.append(row['macd_momentum'])

        data['position'] = positions

    def macd_derivatives(self, roc_window=3):
        self.macd_simple()
        data = self.data
        # Calculate the first derivative of MACD
        data['macd_derivative'] = data['macd'].diff()
        # Calculate the rate of change (ROC) of MACD derivative
        data['macd_derivative_roc'] = data['macd_derivative'].diff(roc_window) / roc_window
        # Initialize Signal column with zeros
        data['signal'] = 0

        # Generate buy (1) and sell (-1) signals based on momentum reflected by ROC
        data.loc[(data['macd_derivative'] > 0) & (
                    data['macd_derivative_roc'] > 0), 'signal'] = 1  # Buy signal with increasing momentum
        data.loc[(data['macd_derivative'] < 0) & (
                    data['macd_derivative_roc'] < 0), 'signal'] = -1  # Sell signal with decreasing momentum

        # Drop intermediate columns
        data.dropna(subset=['macd_derivative', 'macd_derivative_roc'], inplace=True)

    def macd_simple(self):
        short_window, long_window, signal_window = 5, 10, 3
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

    def signal(self):
        self.macd_momentum()
