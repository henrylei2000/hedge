import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import lunardate


import lunardate
from datetime import datetime


def gregorian_to_lunar_day(year, month, day):
    lunar_date = lunardate.LunarDate.fromSolarDate(year, month, day)
    return lunar_date.day

def analyze_seasonality(stock_symbol, start_date='1973-01-01', end_date='2023-01-01'):
    # Download historical stock price data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Check if data is fetched properly
    if stock_data.empty:
        print("No data fetched for the given stock symbol and date range.")
        return

    # Use the 'Close' price for the analysis
    stock_data['Date'] = stock_data.index
    stock_data = stock_data[['Date', 'Close']].copy()

    # Set the 'Date' column as the index
    stock_data.set_index('Date', inplace=True)

    # Calculate daily returns
    stock_data.loc[:, 'Return'] = stock_data['Close'].pct_change()

    # Resample to monthly and weekly data
    monthly_data = stock_data['Return'].resample('ME').mean()
    weekly_data = stock_data['Return'].resample('W').mean()

    # Aggregate data to find average monthly and daily patterns
    stock_data['Month'] = stock_data.index.month
    stock_data['Weekday'] = stock_data.index.weekday
    stock_data['MonthHalf'] = stock_data.index.day <= 15
    stock_data['Week'] = stock_data.index.isocalendar().week
    stock_data['Day'] = stock_data.index.day
    # Add lunar day column
    stock_data['LunarDay'] = stock_data.index.to_series().apply(
        lambda x: gregorian_to_lunar_day(x.year, x.month, x.day))

    whole_monthly_patterns = stock_data.groupby('Month')['Return'].mean()
    monthly_patterns = stock_data.groupby(['Month', 'MonthHalf'])['Return'].mean().unstack()
    weekly_patterns = stock_data.groupby('Weekday')['Return'].mean()
    yearly_weekly_patterns = stock_data.groupby('Week')['Return'].mean()
    daily_patterns = stock_data.groupby('Day')['Return'].mean()
    # Group by lunar days
    lunar_day_patterns = stock_data.groupby('LunarDay')['Return'].mean()

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(stock_data['Close'].dropna(), model='multiplicative', period=365)

    # Plot the decomposition
    plt.figure(figsize=(14, 10))
    plt.subplot(511)
    plt.plot(stock_data['Close'], label='Original')
    plt.legend(loc='best')
    plt.subplot(512)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(513)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(514)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='best')
    plt.subplot(515)
    plt.plot(stock_data['Return'], label='Daily Returns')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot monthly patterns with two halves
    monthly_patterns.plot(kind='bar', figsize=(14, 5))
    plt.title('Average Monthly Returns (Split by First and Second Half)')
    plt.xlabel('Month')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.legend(['First Half', 'Second Half'])
    plt.grid(True)
    plt.show()

    # Plot monthly patterns
    plt.figure(figsize=(14, 5))
    whole_monthly_patterns.plot(kind='bar')
    plt.title('Average Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(12),
               labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.grid(True)
    plt.show()

    # Plot weekly patterns
    plt.figure(figsize=(14, 5))
    weekly_patterns.plot(kind='bar')
    plt.title('Average Weekly Returns')
    plt.xlabel('Weekday')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
    plt.grid(True)
    plt.show()

    # Plot yearly weekly patterns
    plt.figure(figsize=(14, 5))
    yearly_weekly_patterns.plot(kind='bar')
    plt.title('Average Weekly Returns (Split by 52 Weeks)')
    plt.xlabel('Week of the Year')
    plt.ylabel('Average Return')
    plt.grid(True)
    plt.show()

    # Plot daily patterns
    plt.figure(figsize=(14, 5))
    daily_patterns.plot(kind='bar')
    plt.title('Average Daily Returns')
    plt.xlabel('Day of the Month')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(1, 32), labels=[str(i) for i in range(1, 32)], rotation=45)
    plt.grid(True)
    plt.show()

    # Plot lunar day patterns
    plt.figure(figsize=(14, 5))
    lunar_day_patterns.plot(kind='bar')
    plt.title('Average Returns Grouped by Lunar Days')
    plt.xlabel('Lunar Day')
    plt.ylabel('Average Return')
    plt.xticks(ticks=range(1, 31), labels=[str(i) for i in range(1, 31)], rotation=45)
    plt.grid(True)
    plt.show()


    return {
        'decomposition': decomposition,
        'whole_monthly_patterns': whole_monthly_patterns,
        'monthly_patterns': monthly_patterns,
        'weekly_patterns': weekly_patterns,
        'yearly_weekly_patterns': yearly_weekly_patterns,
        'daily_patterns': daily_patterns,
        'lunar_day_patterns': lunar_day_patterns
    }

# Example usage
result = analyze_seasonality(stock_symbol='^GSPC', start_date='1954-07-01', end_date='2024-07-01')
print(result)
