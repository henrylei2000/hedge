import yfinance as yf
import datetime

# Set the start and end dates for the data retrieval
start_date = datetime.datetime.today() - datetime.timedelta(days=30)
end_date = datetime.datetime.today()

# Fetch the stock data for Amazon
amzn = yf.Ticker("AMZN")

# Retrieve the historical stock prices for Amazon over the past month
hist = amzn.history(start=start_date, end=end_date)

# Print the historical stock prices to the console
print(hist)
