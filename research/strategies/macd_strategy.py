from strategy import Strategy
from collections import deque
from scipy.signal import find_peaks
import numpy as np


class MACDStrategy(Strategy):

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

        return approaching_min_boundary, approaching_max_boundary

    def macd_simple(self):
        data = self.data
        data['rolling_strength'] = data['strength'].ewm(span=5, adjust=False).mean()
        data['rolling_macd'] = data['macd'].rolling(window=12).mean()

        data['macd_derivative'] = data['macd'].diff()
        data['rolling_macd_derivative'] = data['macd_derivative'].rolling(window=5).mean()

        data['rolling_rsi'] = data['rsi'].rolling(window=12).mean()
        data['rsi_derivative'] = data['rsi'].diff()
        data['rolling_rsi_derivative'] = data['rsi_derivative'].rolling(window=5).mean()
        data['rolling_vpt'] = data['vpt'].rolling(window=12).mean()

        data['rolling_obv'] = data['obv'].rolling(window=12).mean()

        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data.loc[data['macd'] > data['macd_signal'], 'signal'] = 1
        data.loc[data['macd'] < data['macd_signal'], 'signal'] = -1

        # data.dropna(subset=['close', 'macd', 'macd_derivative', 'rolling_macd', 'macd_signal', 'rsi'], inplace=True)

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

    def linear_regression(self, index, column, window=9, step=1):
        # Predefine x values
        ordinal_index = self.data.index.get_loc(index)
        indices = []
        pos = None
        recent_rows = self.data.loc[:index].tail(window * step)

        if ordinal_index < window * step:
            pos = ordinal_index // step
        else:
            pos = window
            # Start from the last index and iterate backwards with a step size of 2

        for i in range(-(pos * step) + 1, 0, step):
            indices.append(i)
        # Select the rows based on the calculated indices
        selected_rows = recent_rows.iloc[indices]
        x_values = np.array(indices)
        # Initialize list to store y values
        y_values = []

        # Get the last 5 rows of the dataframe
        # Iterate through the last 5 rows
        for i, row in selected_rows.iterrows():
            # Extract the 'close' price as y value
            y = row[column]

            # Append y value to the list
            y_values.append(y)

        # Convert list to numpy array
        y_values = np.array(y_values)
        # Perform linear regression
        X = np.column_stack([x_values, np.ones(x_values.shape[0])])
        coefficients = np.linalg.lstsq(X, y_values, rcond=None)[0]
        # print(f"{column} {coefficients}")
        return coefficients

    def linear(self, points):
        x_values, y_values = [], []
        for p in points[-3:]:
            x, y = p[0], p[1]
            # Append y value to the list
            y_values.append(y)
            x_values.append(x)

        # Convert list to numpy array
        x_values, y_values = np.array(x_values), np.array(y_values)
        # Perform linear regression
        X = np.column_stack([x_values, np.ones(x_values.shape[0])])
        coefficients = np.linalg.lstsq(X, y_values, rcond=None)[0]
        return coefficients[0]

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
            # macd_significance = self.detect_significance(index, 'rolling_macd')
            price_significance = self.detect_significance(index, 'close')
            if len(prev_macd) > 1:
                if price_significance[0]:
                    # bearish, almost bottom
                    if prev_macd[-2] < prev_macd[-1] < macd < 0:
                        position = 1
                    # bearish, just proven
                    if macd < prev_macd[-1] < prev_macd[-2] < 0:
                        position = -1

                if price_significance[1]:
                    # bullish, just proven
                    if macd > prev_macd[-1] > prev_macd[-2] > 0:
                        position = 1
                    # bullish, almost top
                    if prev_macd[-2] > prev_macd[-1] > macd > 0:
                        position = -1

            positions.append(position)
            prev_macd.append(macd)
            prev_strength.append(strength)
            prev_rsi.append(rsi)
            prev_macd_derivatives.append(macd_derivative)

        data['position'] = positions

    def peaks_valleys(self, index=None, column="rsi"):

        # Extract RSI values from the DataFrame and convert to a list for faster access
        if not index:
            data = self.data
        else:
            data = self.data.loc[:index]

        # Initialize variables
        peaks, valleys = [], []
        values = data[column].tolist()

        # First, detect all peaks and valleys
        for i in range(2, len(values) - 2):
            # Read surrounding RSI values to determine peaks or valleys
            prev_1, prev_2 = values[i - 1], values[i - 2]
            curr = values[i]
            next_1, next_2 = values[i + 1], values[i + 2]

            # Check for a peak
            if curr == 99:
                if next_1 < 99:
                    peaks.append((i, curr, 'peak'))

            elif curr == 1:
                if next_1 > 1:
                    valleys.append((i, curr, 'valley'))

            elif curr > prev_1 and curr > prev_2 and curr > next_1 and curr > next_2:
                peaks.append((i, curr, 'peak'))

            elif curr < prev_1 and curr < prev_2 and curr < next_1 and curr < next_2:
                valleys.append((i, curr, 'valley'))

        return peaks, valleys

    def macd_x_rsi(self):
        self.macd_simple()
        self.normalized('macd')

        data = self.data
        previous = deque(maxlen=3)  # Keep track of the last 30 signals
        positions = []  # Store updated signals
        tops_found, bottoms_found = 0, 0
        # Initialize Signal column with zeros
        data['position'] = 0
        count = 0
        for index, row in data.iterrows():
            position = 0
            macd, rsi = row['normalized_macd'], row['rolling_rsi']
            rsis = self.peaks_valleys(index, 'rolling_rsi')
            macds = self.peaks_valleys(index, 'normalized_macd')

            if len(macds) > 10 and len(rsis) > 10:
                # RSI Lifting MACD
                # strength & velocity (interval between peaks and valleys)
                macd_points, rsi_points = [], []
                for macd_index, macd_value, macd_type in reversed(macds[-4:]):
                    driving_rsi = [(i, v, t) for i, v, t in rsis[-10:] if i <= macd_index and (t == macd_type or macd_value == 100 or macd_value == 0)]
                    if len(driving_rsi) > 1:
                        macd_points.append((macd_index, macd_value, macd_type))
                        rsi_points.append(driving_rsi[-2:])

                """
                process macd and rsi signals
                - for each macd
                    - peak / valley
                    - absolute value (positive / negative)
                    - change ratio (since last valley / peak)
                    - velocity (peak-valley interval)
                - for each rsi
                    - peak / valley
                    - absolute value (positive / negative)
                    - change ratio (since last valley / peak)
                    - velocity (peak-valley interval)
                - historical regression (?)
                
                Theories
                - macd resilience to rsi
                    - price will be following the trend of macd
                """

                if len(macd_points):
                    # assumption: NO consecutive peaks and valleys of macd
                    if macd_points[0][2] == 'valley':  # to buy
                        if macd_points[0][1] < rsi_points[0][-1][1] < 40:
                            if rsi > 50:
                                position = 1
                            elif rsi < 20:
                                position = -1
                    if macd_points[0][2] == 'peak':  # to sell
                        if macd_points[0][1] > rsi_points[0][-1][1] > 60:
                            if rsi < 50:
                                position = -1
                            elif rsi > 80:
                                position = 1

            positions.append(position)
            count += 1
        data['position'] = positions

    def zero_crossing(self):
        self.macd_simple()
        data = self.data
        previous = deque(maxlen=3)  # Keep track of the last 3 signals
        positions = []  # Store updated signals
        hold = False
        # Initialize Signal column with zeros
        data['position'] = 0

        for index, row in data.iterrows():
            position = 0
            current = row['strength']
            if len(previous):
                if previous[-1] < 0 < current and not hold:
                    position = 1
                    hold = True
                if previous[-1] > 0 > current and hold:
                    position = -1
                    hold = False
            positions.append(position)
            previous.append(current)

        data['position'] = positions
        self.snapshot([100, 200], ['strength', 'macd'])

    def crossover(self):
        self.macd_simple()
        data = self.data
        position = 0
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        hold = False
        count = 0
        for index, row in data.iterrows():
            mpeaks, mvalleys = self.peaks_valleys(index, 'rolling_macd')
            if hold and row['obv'] < row['rolling_obv'] and row['macd'] < row['rolling_macd'] and row['macd'] < 0 and row['rsi'] > 25:
                if len(mpeaks):
                    if (len(mvalleys) and mvalleys[-1][0] < mpeaks[-1][0]) or len(mvalleys) == 0:
                        position = -1
                        hold = False
            if not hold and row['obv'] > row['rolling_obv'] and row['macd'] > row['rolling_macd'] and row['macd'] > 0 and row['rsi'] < 75:
                if len(mvalleys):
                    if (len(mpeaks) and mpeaks[-1][0] < mvalleys[-1][0]) or len(mpeaks) == 0:
                        position = 1
                        hold = True

            positions.append(position)
            count += 1
        data['position'] = positions

    def trend(self):
        self.macd_simple()
        data = self.data
        position = 0
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        hold = False
        count = 0
        for index, row in data.iterrows():
            visible_rows = data.loc[:index]  # recent rows
            prices = visible_rows['close']

            # Identify peaks and valleys
            peaks, _ = find_peaks(prices, distance=1, prominence=0.1)
            valleys, _ = find_peaks(-prices, distance=1, prominence=0.1)

            # Perform linear regression on peaks
            if len(peaks) > 5:
                peak_indices = np.array(peaks)
                peak_prices = prices.iloc[peaks]
                a_peaks, b_peaks = np.polyfit(peak_indices, peak_prices, 1)
                a_recent, b_recent = np.polyfit(peak_indices[-3:], peak_prices[-3:], 1)
                if a_peaks * a_recent < 0:
                    if a_recent > a_peaks:
                        position = 1
                    print(f"[{a_peaks:.3f} {a_recent:.3f}] [{b_peaks:.3f} {b_recent:.3f}] @{peak_indices[-1]} {data.index.get_loc(index)} {index}")

            # Perform linear regression on valleys
            if len(valleys) > 5:
                valley_indices = np.array(valleys)
                valley_prices = prices.iloc[valleys]
                a_valleys, b_valleys = np.polyfit(valley_indices, valley_prices, 1)
                print(f"[{a_valleys:.3f} {b_valleys:.3f}]")

            positions.append(position)
            count += 1
        data['position'] = positions

    def rsi(self):
        self.macd_simple()
        self.normalized('rolling_macd')
        data = self.data
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0
        hold = False
        count = 0
        for index, row in data.iterrows():

            position = 0
            mpeaks, mvalleys = self.peaks_valleys(index, 'normalized_rolling_macd')
            rpeaks, rvalleys = self.peaks_valleys(index, 'rolling_rsi')

            rsi = row['rolling_rsi']
            macd = row['normalized_rolling_macd']


            """
            process macd and rsi signals
            - for each macd
                - peak / valley
                - absolute value (positive / negative)
                - change ratio (since last valley / peak)
                - velocity (peak-valley interval)
            - for each rsi
                - peak / valley
                - absolute value (positive / negative)
                - change ratio (since last valley / peak)
                - velocity (peak-valley interval)
            - historical regression (?)

            Theories
            - macd resilience to rsi
                - price will be following the trend of macd
            
            Scenarios (status / trend):
                - high macd, high rsi: already top?
                - high macd, low rsi (valley): resistence?
                - low macd, low rsi: already bottom?
                = low macd, high rsi (?)
            """

            """
            buy: 1) rsi_Valley < 20 spotted; 2) bounce back to 50 quickly; 3) macd_valley > rsi_valley spotted
            sell: 1) rsi_peak > 80 spotted; 2) drop to 50 quickly; 3) macd_peak < rsi_peak spotted
            sell: 1) rsi_peak < 50 spotted after a buy; 2) macd < 50 before a peak appears
            """
            # if len(rvalleys) and len(mvalleys):
            #     if rvalleys[-1][1] < 10 and mvalleys[-1][0] > rvalleys[-1][0]:
            #         print(f"{count} {index} {rvalleys[-1]} {mvalleys[-1]} rsi {rsi} {row['close']:.3f}")

            if not hold and len(rvalleys) and len(mvalleys):  # searching for a buying opportunity - bullish signal
                # rsi trend reversal
                if rsi - rvalleys[-1][1] > 35 and rvalleys[-1][1] < 15:
                    # sharpness of the reversal
                    if count - rvalleys[-1][0] < 10:
                        # relation to macd (following rsi closely and a reliable base)
                        if 0 < mvalleys[-1][0] - rvalleys[-1][0] < 5 and mvalleys[-1][1] > 35:
                            # volume consideration
                            if row['obv'] > row['rolling_obv'] or True:
                                position = 1
                                hold = True
            elif hold and len(mpeaks) and len(rpeaks):  # waiting for a selling opportunity - bearish signal
                # trend reversal
                if rpeaks[-1][1] - rsi > 30 and rpeaks[-1][1] > 70 and count - rpeaks[-1][0] < 10 and 0 < mpeaks[-1][0] - rpeaks[-1][0] < 5:
                    if row['obv'] < row['rolling_obv'] or True:
                        position = -1
                        hold = False

                # a sudden drop of macd
                elif mpeaks[-1][1] - macd > 40 and 0 < count - mpeaks[-1][0] < 2:
                    if row['obv'] < row['rolling_obv'] or True:
                        position = -1
                        hold = False

            positions.append(position)
            count += 1

        data['position'] = positions

    def wave(self):
        self.macd_simple()
        data = self.data
        previous = deque(maxlen=3)  # Keep track of the last 30 signals
        wait = 1
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0

        for index, row in data.iterrows():
            position = 0
            current = row['rolling_macd']
            waves = self.wave_sums('rolling_macd', index)

            if len(waves) > 1 and ((current > 0) == (waves[-1] > 0)) and len(previous) == 3:
                # bullish, just proven
                if waves[-1] > 0:
                    if current > previous[-1] > previous[-2]:
                        position = 1
                # bearish, almost bottom
                if waves[-1] < -abs(waves[-2]) * 0.618:
                    if previous[-2] < previous[-1] < current:
                        position = 1
                # bullish, almost top
                if waves[-1] > abs(waves[-2]) * 0.618:
                    if previous[-2] > previous[-1] > current:
                        position = -1
                # bearish, just proven
                if waves[-1] < 0:
                    if current < previous[-1] < previous[-2]:
                        position = -1

            positions.append(position)
            previous.append(current)

        data['position'] = positions

    def wave_sums(self, column, index=None, threshold=9):

        if not index:
            data = self.data
        else:
            data = self.data.loc[:index]

        # Initialize variables
        wave_sums = []
        current_wave_sum = 0
        current_wave_sign = None
        current_wave_length = 0

        # Iterate through DataFrame rows
        for _, row in data.iterrows():
            value = row[column]

            # Update current wave state
            if current_wave_sign is None or (current_wave_sign > 0) == (value > 0):
                current_wave_sum += value
                current_wave_length += 1
            else:
                if current_wave_length > threshold:
                    wave_sums.append(current_wave_sum)
                current_wave_sum, current_wave_length = value, 1
            current_wave_sign = value > 0

        # Append the last wave sum if the last wave has more than 1 value
        if current_wave_length > threshold:
            wave_sums.append(current_wave_sum)

        # Merge consecutive wave sums with the same sign
        merged_wave_sums = []
        for wave_sum in wave_sums:
            if not merged_wave_sums or (merged_wave_sums[-1] > 0) == (wave_sum > 0):
                if merged_wave_sums:
                    merged_wave_sums[-1] += wave_sum
                else:
                    merged_wave_sums.append(wave_sum)
            else:
                merged_wave_sums.append(wave_sum)

        return merged_wave_sums

    def detect_abnormality(self, index, column, prominence=0.00382):
        # Calculate the minimum and maximum values within the recent window
        recent_rows = self.data.loc[:index]

        # Find peaks (tops) and valleys (bottoms)
        peaks, _ = find_peaks(recent_rows[column], prominence=prominence)  # Adjust prominence as needed
        valleys, _ = find_peaks(-recent_rows[column], prominence=prominence)  # Adjust prominence as needed

        return peaks, valleys

    def abnormality(self):
        self.macd_simple()
        data = self.data
        positions = []  # Store updated signals

        # Initialize Signal column with zeros
        data['position'] = 0

        previous_peaks = set()
        previous_valleys = set()

        for index, row in data.iterrows():
            position = 0
            current = row['rolling_macd']
            peaks, valleys = self.detect_abnormality(index, 'strength')
            ordinal_index = self.data.index.get_loc(index)
            # Convert peaks and valleys to sets
            peaks = set(peaks)
            valleys = set(valleys)
            # Print new peaks that haven't been seen before
            new_peaks = peaks - previous_peaks
            for peak in new_peaks:
                # print(f"PEAK {data.iloc[peak]['close']:.2f} {peaks} - {peak}, @{index} [{ordinal_index}, {row['close']:.2f}]")
                if ordinal_index - peak < 5:
                    position = -1

            # Print new valleys that haven't been seen before
            new_valleys = valleys - previous_valleys
            for valley in new_valleys:
                # print(f"VALLEY {data.iloc[valley]['close']:.2f} {valleys} - {valley}, @{index} [{ordinal_index}, {row['close']:.2f}]")
                if ordinal_index - valley < 5:
                    position = 1
            # Update previous peaks and valleys
            previous_peaks.update(new_peaks)
            previous_valleys.update(new_valleys)

            positions.append(position)

        data['position'] = positions

    def signal(self):
        self.zero_crossing()
        # waves = self.wave_sums('strength', '2024-03-26 12:59')
        # print(waves)
        #
        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(waves)), waves, color='skyblue')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title(f"{self.symbol} Bar Chart")
        # plt.show()
        #
        # print(sum(waves), len(waves))