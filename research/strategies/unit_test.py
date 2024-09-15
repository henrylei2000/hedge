import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import configparser


def calculate_macd(data):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the given data.

    Parameters:
    - data: A DataFrame containing 'Close' prices.

    Returns:
    - A DataFrame containing the MACD line and the Signal line.
    """
    # Calculate the 12-day and 26-day EMAs
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    macd = ema_12 - ema_26

    # Calculate the Signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Create a DataFrame to hold the MACD and Signal lines
    macd_df = pd.DataFrame({
        'MACD': macd,
        'Signal': signal
    })

    return pd.Series(macd, index=data.index)

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

def calculate_volume(data):
    return pd.Series(data['Volume'], index=data.index)

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


def analyze_trends(ticker='TQQQ', d_day='2024-07-12', distance=3, interval='1m'):
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
    if interval == '1d':
        ticker = yf.download(ticker, start=pd.to_datetime(d_day, format='%Y-%m-%d') - pd.DateOffset(months=12), end=d_day, interval=interval)
        prices = ticker['Close']

    if interval == '1m':
        # Load Alpaca API credentials from configuration file
        config = configparser.ConfigParser()
        config.read('config.ini')
        # Access configuration values
        api_key = config.get('settings', 'API_KEY')
        secret_key = config.get('settings', 'SECRET_KEY')
        # Initialize Alpaca API
        api = tradeapi.REST(api_key, secret_key, api_version='v2')

        # Retrieve stock price data from Alpaca
        start_time = pd.Timestamp(d_day + ' 09:30', tz='America/New_York').tz_convert('UTC')
        end_time = pd.Timestamp(d_day + ' 16:00', tz='America/New_York').tz_convert('UTC')
        ticker = api.get_bars(ticker, '5Min', start=start_time.isoformat(), end=end_time.isoformat()).df

        if not ticker.empty:
            # Convert timestamp index to Eastern Timezone (EST)
            ticker.index = ticker.index.tz_convert('US/Eastern')
            # Filter rows between 9:30am and 4:00pm EST
            data = ticker.between_time('9:30', '16:00')
            if not ticker.empty:
                ticker['Close'] = ticker['close']
                ticker['Volume'] = ticker['volume']
                prices = data['close']
            else:
                prices = []

    prominence = data.iloc[0]['close']  * 0.00169 + 0.003
    # prominence = prices.iloc[-1] * 0.00125 + 0.005
    print(f"----------- {prominence}")
    # Identify peaks and valleys
    peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
    print(peaks)
    print(valleys)

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
    # obv = calculate_obv(ticker)
    obv = calculate_volume(ticker)
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
    ax1.set_title(f"{d_day} Stock Price Analysis")
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


def predict_next_day_peak_valley(ticker='TQQQ', next_day='2024-07-30', months=6):
    """
    Predicts the next day's peak and valley based on one year of daily prices and linear regression.

    Parameters:
    - ticker: The stock ticker symbol (default is 'TQQQ').

    Returns:
    - A dictionary containing the predicted peak and valley prices for the next day.
    """
    end_date = pd.to_datetime(next_day, format='%Y-%m-%d')
    start_date = end_date - pd.DateOffset(months=months)

    # Download one year of daily price data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    prices = data['Close']
    volumes = data['Volume']

    # Calculate OBV
    obv = calculate_obv(data)

    # Identify peaks and valleys
    prominence = data.iloc[0]['Close'] * 0.00169 + 0.003
    peaks, _ = find_peaks(prices, distance=10, prominence=0.1)
    valleys, _ = find_peaks(-prices, distance=10, prominence=0.1)

    # Perform linear regression on peaks
    peak_indices = np.array(peaks)
    peak_prices = prices.iloc[peaks]
    a_peaks, b_peaks = linear_regression(peak_indices, peak_prices)

    # Perform linear regression on valleys
    valley_indices = np.array(valleys)
    valley_prices = prices.iloc[valleys]
    a_valleys, b_valleys = linear_regression(valley_indices, valley_prices)

    # Predict the next day's peak and valley
    next_day_index = len(prices)
    predicted_peak = a_peaks * next_day_index + b_peaks
    predicted_valley = a_valleys * next_day_index + b_valleys

    # Determine the type of the last recognized abnormal day
    last_peak_index = peak_indices[-1] if len(peak_indices) > 0 else -1
    last_valley_index = valley_indices[-1] if len(valley_indices) > 0 else -1

    if last_peak_index > last_valley_index:
        last_abnormal_day = 'Peak'
    else:
        last_abnormal_day = 'Valley'

    # Indicate whether the next day is predicted to be a peak or a valley based on the last abnormal day
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

    # Create a new index for the next day

    next_day_date = prices.index[-1] + pd.DateOffset(days=1)
    extended_index = prices.index.append(pd.Index([next_day_date]))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot price data and linear regression lines
    ax1.plot(prices, label='Price', color='blue')
    ax1.plot(prices.index[peaks], prices.iloc[peaks], 'ro', label='Peaks')
    ax1.plot(prices.index[valleys], prices.iloc[valleys], 'go', label='Valleys')
    ax1.plot(prices.index, a_peaks * np.arange(len(prices)) + b_peaks, 'r--', label='Peaks Linear Fit')
    ax1.plot(prices.index, a_valleys * np.arange(len(prices)) + b_valleys, 'g--', label='Valleys Linear Fit')
    ax1.axvline(next_day_date, color='purple', linestyle='--', label='Next Day')
    ax1.axhline(predicted_peak, color='red', linestyle='--', label='Predicted Peak')
    ax1.axhline(predicted_valley, color='green', linestyle='--', label='Predicted Valley')
    for peak in peaks[-2:]:
        ax1.text(prices.index[peak], prices.iloc[peak], prices.index[peak].strftime('%Y-%m-%d'),
                 horizontalalignment='left', size='small', color='red', weight='semibold')
    for valley in valleys[-2:]:
        ax1.text(prices.index[valley], prices.iloc[valley], prices.index[valley].strftime('%Y-%m-%d'),
                 horizontalalignment='left', size='small', color='green', weight='semibold')

    ax1.set_title(f'{ticker} Price Analysis with Predicted Peaks and Valleys {prices.index[-1]}')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot OBV data
    ax2.plot(volumes, label='volume', color='purple')
    ax2.set_title('volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('volume')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return prediction


ticker = 'TQQQ'
d_day = '2023-03-09'
prediction = predict_next_day_peak_valley(ticker, d_day, months=12)
print(prediction)
trends = analyze_trends(ticker, d_day, distance=5)