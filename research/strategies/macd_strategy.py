from strategy import Strategy
from collections import deque
from scipy.signal import find_peaks


class MACDStrategy(Strategy):

    # Function to calculate wave sums
    def wave_sums(self, column):
        # Initialize variables
        wave_sums = []
        current_wave_sum = 0
        current_wave_sign = None

        # Sample DataFrame with alternating positive and negative values
        # import pandas as pd
        # data = pd.DataFrame({'values': [-1, -3, 2, -1, -5, -4, 2, 3, 4, 6, 4, 1, -1, -6, -3, -4, -2]})
        # column = 'values'
        # Iterate through DataFrame rows
        for index, row in self.data.iterrows():
            value = row[column]

            # Check if the current value has the same sign as the previous value
            if current_wave_sign is None or (current_wave_sign > 0) == (value > 0):
                # Accumulate value within the current wave
                current_wave_sum += value
            else:
                # Start a new wave
                wave_sums.append(current_wave_sum)
                current_wave_sum = value

            # Update the current wave sign
            current_wave_sign = value > 0

        # Append the last wave sum
        wave_sums.append(current_wave_sum)

        return wave_sums

    def detect_significance(self, index, column, window=39, boundary_ratio=0.1):
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

        # Find peaks (tops) and valleys (bottoms)
        peaks, _ = find_peaks(recent_rows[column], prominence=boundary_ratio)  # Adjust prominence as needed
        valleys, _ = find_peaks(-recent_rows[column], prominence=boundary_ratio)  # Adjust prominence as needed

        tops = recent_rows.iloc[peaks]
        bottoms = recent_rows.iloc[valleys]

        return approaching_min_boundary, approaching_max_boundary

    def macd_simple(self):
        short_window, long_window, signal_window = 19, 39, 9   # 3, 7, 2
        dataset = [self.data]
        if self.reference:
            dataset += [self.qqq, self.spy, self.dia]
        for data in dataset:
            # Calculate short-term and long-term exponential moving averages
            data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
            data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()

            # Calculate MACD line
            data['macd'] = data['short_ma'] - data['long_ma']
            # Calculate Signal line
            data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
            data['strength'] = data['macd'] - data['signal_line']
            data['rolling_strength'] = data['strength'].ewm(span=5, adjust=False).mean()
            data['rolling_macd'] = data['macd'].rolling(window=5).mean()

            data['momentum'] = data['macd'] - data['strength']

            # Calculate the first derivative of MACD
            data['macd_derivative'] = data['macd'].diff()
            data['rolling_macd_derivative'] = data['macd_derivative'].rolling(window=5).mean()

            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            data['rolling_rsi'] = data['rsi'].rolling(window=5).mean()
            # Calculate the first derivative of MACD
            data['rsi_derivative'] = data['rsi'].diff()
            data['rolling_rsi_derivative'] = data['rsi_derivative'].rolling(window=5).mean()

            # Generate Buy and Sell signals
            data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

            # Buy signal: MACD crosses above Signal line
            data.loc[data['macd'] > data['signal_line'], 'signal'] = 1

            # Sell signal: MACD crosses below Signal line
            data.loc[data['macd'] < data['signal_line'], 'signal'] = -1

            data.dropna(subset=['close', 'macd', 'macd_derivative', 'rolling_macd', 'signal_line', 'rsi'], inplace=True)

            data.to_csv(f"{self.symbol}.csv")

    def significance_reference(self):
        self.macd_simple()
        data = self.data
        prev_macd_derivatives = deque(maxlen=3)  # Keep track of the last 30 signals
        prev_macd_strength = deque(maxlen=3)
        wait = 3
        scale = 2
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0

        for index, row in data.iterrows():
            position = 0
            strength, macd_derivative = row['macd'] - row['signal_line'], row['macd_derivative']

            if len(prev_macd_derivatives) >= wait:
                significance = self.detect_significance(index, 'close')
                strength_significance = self.detect_significance(index, 'macd_strength')

                if significance[1] or strength_significance[1]:
                    print(f"PEAK {row['close']:.2f} @{index} {significance} {prev_macd_derivatives[-1]:.3f} > {macd_derivative:.3f} > 0")
                    if prev_macd_derivatives[-1] > macd_derivative * scale > 0:
                        print(f"QQQ: {self.qqq.loc[index]['macd']} < SPY: {self.spy.loc[index]['macd']} < DIA: {self.dia.loc[index]['macd']}")
                        if self.qqq.loc[index]['macd_derivative'] < self.spy.loc[index]['macd_derivative']:
                            position = -1

                if significance[0] or strength_significance[0]:
                    print(f"VALLEY {row['close']:.2f} @{index} {significance} {prev_macd_derivatives[-1]:.3f} < {macd_derivative:.3f} < 0")
                    if prev_macd_derivatives[-1] < macd_derivative * scale < 0:
                        print(f"QQQ: {self.qqq.loc[index]['macd']} < SPY: {self.spy.loc[index]['macd']} < DIA: {self.dia.loc[index]['macd']}")
                        if self.qqq.loc[index]['macd_derivative'] > self.spy.loc[index]['macd_derivative']:
                            position = 1

            positions.append(position)
            prev_macd_derivatives.append(row['macd_derivative'])
            prev_macd_strength.append(row['macd_strength'])

        data['position'] = positions

        data.to_csv(f"{self.symbol}.csv")

    def significance(self):
        self.macd_simple()
        data = self.data
        prev_macd_derivatives = deque(maxlen=3)  # Keep track of the last 30 signals
        prev_macd = deque(maxlen=3)
        prev_strength = deque(maxlen=3)
        prev_rsi = deque(maxlen=3)
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0

        for index, row in data.iterrows():
            position = 0
            macd, rsi, macd_derivative, strength = row['rolling_macd'], row['rolling_rsi'], row['macd_derivative'], row['rolling_strength']
            significance = self.detect_significance(index, 'rolling_macd')
            price_significance = self.detect_significance(index, 'close')
            if len(prev_strength):
                if significance[0] and price_significance[0]:
                    position = 1
                if significance[1] and price_significance[1]:
                    position = -1

            positions.append(position)
            prev_macd.append(macd)
            prev_strength.append(strength)
            prev_rsi.append(rsi)
            prev_macd_derivatives.append(macd_derivative)

        data['position'] = positions
        data.to_csv(f"{self.symbol}.csv")

    def zero_crossing(self):
        self.macd_simple()
        data = self.data
        previous = deque(maxlen=3)  # Keep track of the last 30 signals
        wait = 1
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0

        for index, row in data.iterrows():
            position = 0
            current = row['strength']

            if len(previous) == 0:
                if current > 0:
                    position = 1
            else:

                if previous[-1] > 0 > current:
                    position = -1

                if previous[-1] < 0 < current:
                    position = 1

            positions.append(position)
            previous.append(current)

        data['position'] = positions

        data.to_csv(f"{self.symbol}.csv")

    def signal(self):
        self.significance()
        waves = self.wave_sums('strength')
        print(waves)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(waves)), waves, color='skyblue')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f"{self.symbol} Bar Chart")
        plt.show()


        print(sum(waves), len(waves))