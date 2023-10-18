import yfinance as yf
import pandas as pd

# Function to calculate the correlation coefficient
def calculate_correlation(ticker1, ticker2, start_date, end_date):
    try:
        # Download historical stock data for both tickers within the specified date range
        data1 = yf.download(ticker1, start=start_date, end=end_date)
        data2 = yf.download(ticker2, start=start_date, end=end_date)

        # Ensure both dataframes have the same length and align their indices
        data1 = data1['Adj Close']
        data2 = data2['Adj Close']
        data1, data2 = data1.align(data2, join='inner')

        # Calculate the correlation coefficient
        correlation = data1.corr(data2)

        return correlation

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Input the two stock symbols
ticker1 = input("Enter the first stock symbol (e.g., AAPL): ")
ticker2 = input("Enter the second stock symbol (e.g., MSFT): ")

# Input the start date for the analysis
start_date = input("Enter the start date (YYYY-MM-DD): ")

# Input the end date for the analysis
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Calculate the correlation coefficient
correlation_coefficient = calculate_correlation(ticker1, ticker2, start_date, end_date)

if correlation_coefficient is not None:
    print(f"Correlation coefficient between {ticker1} and {ticker2} from {start_date} to {end_date}: {correlation_coefficient:.2f}")
