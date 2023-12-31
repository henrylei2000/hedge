import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# ... (previous code)

def get_top_gainers_and_decliners(start_date, end_date):
    try:
        # Define a function to calculate percentage change
        def calculate_percentage_change(df):
            return (df['Close'][-1] - df['Close'][0]) / df['Close'][0] * 100

        # Create an empty DataFrame to store the results
        results = pd.DataFrame(columns=['Ticker', 'Percentage Change'])

        # Download historical data for the S&P 500 index
        sp500_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

        # Get the list of S&P 500 component tickers
        sp500_tickers = get_sp500_tickers()

        for ticker, _ in sp500_tickers:
            # Download historical data for the ticker
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if ticker_data.empty:
                continue

            # Calculate percentage change for the ticker
            percentage_change = calculate_percentage_change(ticker_data)

            # Append the result to the DataFrame
            results = results.append({'Ticker': ticker, 'Percentage Change': percentage_change}, ignore_index=True)

        # Sort the results by percentage change
        results = results.sort_values(by='Percentage Change', ascending=False)

        # Get the top 5 gainers and top 5 decliners
        top_gainers = results.head(5)
        top_decliners = results.tail(5)

        return top_gainers, top_decliners

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # Input the start date for the analysis
    start_date = input("Enter the start date (YYYY-MM-DD): ")

    # Input the end date for the analysis
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Get top gainers and decliners
    top_gainers, top_decliners = get_top_gainers_and_decliners(start_date, end_date)

    # Print the results
    print("\nTop 5 Gainers:")
    print(top_gainers)
    print("\nTop 5 Decliners:")
    print(top_decliners)
