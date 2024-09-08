import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf


def predict_with_fft(ticker, start_date, end_date, sampling_rate=1, num_predictions=10):
    """
    Predicts future stock prices using Fourier Transform on detrended data.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    sampling_rate (int): Sampling rate (1 if daily data).
    num_predictions (int): Number of future points to predict.

    Returns:
    None - Displays the original, trend, detrended data, frequency spectrum, and future predictions.
    """
    # Download historical stock prices
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close'].values  # Use closing prices

    # Check if data was downloaded successfully
    if len(prices) == 0:
        print("No data found for the specified ticker and date range.")
        return

    time = np.arange(len(prices)).reshape(-1, 1)  # Time as independent variable for regression

    # Fit a linear regression model to the data
    model = LinearRegression()
    model.fit(time, prices)

    # Predict the trend using the linear model
    trend = model.predict(time)

    # Detrend the data by subtracting the trend
    detrended_prices = prices - trend

    # Calculate the FFT of the detrended price data
    fft_result = np.fft.fft(detrended_prices)

    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(detrended_prices), d=sampling_rate)

    # Get the magnitude of the FFT result (amplitude spectrum)
    magnitude = np.abs(fft_result)

    # Zero out small frequencies (noise filtering)
    threshold = np.mean(np.abs(fft_result)) * 1.5
    fft_filtered = np.where(np.abs(fft_result) > threshold, fft_result, 0)

    # Inverse FFT to reconstruct the detrended signal
    reconstructed_detrended = np.fft.ifft(fft_filtered).real

    # Predict future detrended prices by repeating the identified cycles
    future_detrended = np.tile(reconstructed_detrended, int(np.ceil(num_predictions / len(reconstructed_detrended))))[
                       :num_predictions]

    # Extend the trend line for future predictions
    future_time = np.arange(len(prices), len(prices) + num_predictions).reshape(-1, 1)
    future_trend = model.predict(future_time)

    # Combine trend and cyclical components for future predictions
    future_predictions = future_trend + future_detrended

    # Plot the original data, trend, detrended data, frequency spectrum, and future predictions
    plt.figure(figsize=(14, 10))

    # Original data with trend
    plt.subplot(4, 1, 1)
    plt.plot(prices, label='Original Data')
    plt.plot(trend, label='Trend (Linear Regression)', linestyle='--')
    plt.title(f'{ticker} Stock Price and Trend')
    plt.legend()

    # Detrended data
    plt.subplot(4, 1, 2)
    plt.plot(detrended_prices, label='Detrended Data')
    plt.title('Detrended Data')
    plt.legend()

    # Frequency spectrum
    plt.subplot(4, 1, 3)
    plt.plot(frequencies, magnitude)
    plt.title('Frequency Spectrum of Detrended Stock Price Fluctuations')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Future predictions
    plt.subplot(4, 1, 4)
    plt.plot(range(len(prices)), prices, label='Original Data')
    plt.plot(range(len(prices), len(prices) + num_predictions), future_predictions, label='Future Predictions',
             linestyle='--')
    plt.title('Future Price Predictions Using FFT')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage
predict_with_fft(ticker='TQQQ', start_date='2023-01-01', end_date='2024-10-01', num_predictions=30)
