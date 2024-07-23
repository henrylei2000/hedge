import pandas as pd

def analyze_rsi_series(data):
    if 'rsi' not in data.columns:
        raise ValueError("Data must include 'rsi' column")

    # Initialize variables
    result = []
    rsi_values = data['rsi'].tolist()

    # First, detect all peaks and valleys
    for i in range(2, len(rsi_values) - 2):
        # Read surrounding RSI values to determine peaks or valleys
        prev_rsi_1, prev_rsi_2 = rsi_values[i - 2], rsi_values[i - 1]
        curr_rsi = rsi_values[i]
        next_rsi_1, next_rsi_2 = rsi_values[i + 1], rsi_values[i + 2]

        # Check for a peak
        if curr_rsi > max(prev_rsi_1, prev_rsi_2, next_rsi_1, next_rsi_2):
            print(f"{i} ----- {rsi_values[i]}")
            result.append((i, curr_rsi, 'peak'))
        # Check for a valley
        elif curr_rsi < min(prev_rsi_1, prev_rsi_2, next_rsi_1, next_rsi_2):
            result.append((i, curr_rsi, 'valley'))

    return result

# Example usage:
data = pd.DataFrame({'rsi': [30, 45, 70, 60, 55, 75, 65, 50, 45, 55, 65, 30, 20]})
# print(analyze_rsi_series(data))


def filter_rsi_subset(macd, rsi):
    if len(macd) > 3 and len(rsi) > 3:
        # RSI Lifting MACD
        # strength & velocity (interval between peaks and valleys)
        macd_points, rsi_points = [], []
        print(macd[-4:])
        print(rsi[-10:])
        for macd_index, macd_value, macd_type in reversed(macd[-4:]):
            macd_points.append((macd_index, macd_value, macd_type))
            causing_rsi = []
            for rsi_index, rsi_value, rsi_type in reversed(rsi[-10:]):
                if rsi_index < macd_index or rsi_index == macd_index:
                    if rsi_type != macd_type:
                        if len(causing_rsi):  # conclude the current search
                            break
                    else:
                        causing_rsi.append((rsi_index, rsi_value, rsi_type))
            rsi_points.append(causing_rsi)
        print(f'{macd_points} -------------------------- {rsi_points}')


# Example usage
macd = [(2, 0.005656788341390495, 'valley'), (8, 0.024355482933032135, 'peak'), (11, 0.017572528247562502, 'valley'), (27, 0.10857385642029271, 'peak')]
rsi = [(2, 41.228070175438475, 'valley'), (5, 62.85973947288657, 'peak'), (8, 50.89158345221094, 'valley'), (11, 56.01851851851898, 'peak'), (14, 39.90825688073404, 'valley'), (16, 63.359591228443854, 'peak'), (18, 58.470986869971036, 'valley'), (27, 85.73446327683581, 'peak')]

# filter_rsi_subset(macd, rsi)

def group_by_type(input_list):
    if not input_list:
        return []

    result = []
    current_group = [input_list[0]]

    for i in range(1, len(input_list)):
        if input_list[i][2] != current_group[-1][2]:
            result.append(current_group)
            current_group = [input_list[i]]
        else:
            current_group.append(input_list[i])

    result.append(current_group)
    return result


# Example usage:
input_list = [(1, 1.0, "peak"), (2, 1.5, "valley"), (3, 0.5, "valley"), (7, 1.5, "valley")]
output_list = group_by_type(input_list)
print(output_list)

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def calculate_obv(data):
    """
    Calculates the On-Balance Volume (OBV) for the given data.

    Parameters:
    - data: A DataFrame containing 'Close' and 'Volume' columns.

    Returns:
    - A Series containing the OBV values.
    """
    obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                   np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()
    return pd.Series(obv, index=data.index)

def linear_regression(x, y):
    """
    Performs linear regression on the given x and y data.

    Parameters:
    - x: The x values.
    - y: The y values.

    Returns:
    - A tuple (a, b) representing the slope and intercept of the line.
    """
    a, b = np.polyfit(x, y, 1)
    return a, b


def analyze_trends(start_date='2024-07-11', end_date='2024-07-12', distance=3, prominence=0.5):
    """
    Downloads historical stock price data, identifies peaks and valleys, and analyzes trends.

    Parameters:
    - start_date: The start date for the historical data (format: 'YYYY-MM-DD').
    - end_date: The end date for the historical data (format: 'YYYY-MM-DD').
    - distance: Required minimal horizontal distance (in number of data points) between neighboring peaks.
    - prominence: Required prominence of peaks.

    Returns:
    - A DataFrame with columns 'Price', 'Trend', and 'Type'.
    - Plots the prices with peaks and valleys and the OBV.
    """
    # Download historical data for TQQQ
    ticker = yf.download('TQQQ', start=start_date, end=end_date, interval='1m')
    prices = ticker['Close']

    # Identify peaks and valleys
    peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)

    trends = pd.DataFrame({
        'Price': prices,
        'Trend': np.nan,
        'Type': np.nan
    }, dtype=object)

    # Use iloc to align indices correctly
    trends.iloc[peaks, trends.columns.get_loc('Trend')] = 'Peak'
    trends.iloc[valleys, trends.columns.get_loc('Trend')] = 'Valley'

    # Forward fill the trends to identify uptrends and downtrends
    trends['Trend'] = trends['Trend'].ffill()
    trends['Type'] = np.where(trends['Trend'] == 'Peak', 'Downtrend', 'Uptrend')

    # Calculate OBV
    obv = calculate_obv(ticker)

    # Perform linear regression on peaks
    peak_indices = np.array(peaks)
    peak_prices = prices.iloc[peaks]
    a_peaks, b_peaks = linear_regression(peak_indices, peak_prices)

    # Perform linear regression on valleys
    valley_indices = np.array(valleys)
    valley_prices = prices.iloc[valleys]
    a_valleys, b_valleys = linear_regression(valley_indices, valley_prices)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(prices, label='Price', color='blue')
    ax1.plot(prices.iloc[peaks], 'ro', label='Peaks')
    ax1.plot(prices.iloc[valleys], 'go', label='Valleys')
    ax1.plot(prices.index, a_peaks * np.arange(len(prices)) + b_peaks, 'r--', label='Peaks Linear Fit')
    ax1.plot(prices.index, a_valleys * np.arange(len(prices)) + b_valleys, 'g--', label='Valleys Linear Fit')
    ax1.set_title('TQQQ Stock Price Analysis')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(obv, label='OBV', color='purple')
    ax2.set_title('On-Balance Volume (OBV)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('OBV')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return trends


# Example usage:
trends = analyze_trends(
    start_date='2024-07-22',
    end_date='2024-07-23',
    distance=1,
    prominence=0.1)
print(trends)


