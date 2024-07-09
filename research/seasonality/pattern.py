import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import lunardate


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

    whole_monthly_patterns = stock_data.groupby('Month')['Return'].agg(['mean', 'std', 'count'])
    whole_monthly_patterns['sem'] = whole_monthly_patterns['std'] / np.sqrt(whole_monthly_patterns['count'])
    monthly_patterns = stock_data.groupby(['Month', 'MonthHalf'])['Return'].mean().unstack()

    weekly_patterns = stock_data.groupby('Weekday')['Return'].agg(['mean', 'std', 'count'])
    weekly_patterns['sem'] = weekly_patterns['std'] / np.sqrt(weekly_patterns['count'])

    yearly_weekly_patterns = stock_data.groupby('Week')['Return'].agg(['mean', 'std', 'count'])
    yearly_weekly_patterns['sem'] = yearly_weekly_patterns['std'] / np.sqrt(yearly_weekly_patterns['count'])

    daily_patterns = stock_data.groupby('Day')['Return'].agg(['mean', 'std', 'count'])
    # Calculate the standard error of the mean
    daily_patterns['sem'] = daily_patterns['std'] / np.sqrt(daily_patterns['count'])
    # Group by lunar days
    lunar_day_stats = stock_data.groupby('LunarDay')['Return'].agg(['mean', 'std', 'count'])
    # Calculate the standard error of the mean
    lunar_day_stats['sem'] = lunar_day_stats['std'] / np.sqrt(lunar_day_stats['count'])

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

    # Plot monthly patterns and whole monthly patterns in one figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot monthly patterns with two halves
    monthly_patterns.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Average Monthly Returns (Split by First and Second Half)')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Return')
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            rotation=45)
    axes[0].legend(['First Half', 'Second Half'])
    axes[0].grid(True)

    # Plot whole monthly patterns with error bars
    axes[1].bar(whole_monthly_patterns.index, whole_monthly_patterns['mean'], yerr=whole_monthly_patterns['sem'],
                capsize=5)
    axes[1].set_title('Average Monthly Returns')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Return')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            rotation=45)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot weekly and yearly weekly patterns in one figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot weekly patterns with error bars
    axes[0].bar(weekly_patterns.index, weekly_patterns['mean'], yerr=weekly_patterns['sem'], capsize=5)
    axes[0].set_title('Average Weekly Returns')
    axes[0].set_xlabel('Weekday')
    axes[0].set_ylabel('Average Return')
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
    axes[0].grid(True)

    # Plot yearly weekly patterns with error bars
    axes[1].bar(yearly_weekly_patterns.index, yearly_weekly_patterns['mean'], yerr=yearly_weekly_patterns['sem'], capsize=5)
    axes[1].set_title('Average Weekly Returns (Split by 52 Weeks)')
    axes[1].set_xlabel('Week of the Year')
    axes[1].set_ylabel('Average Return')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot daily and lunar day patterns in one figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot daily patterns
    axes[0].bar(daily_patterns.index, daily_patterns['mean'], yerr=daily_patterns['sem'], capsize=5)
    axes[0].set_title('Average Daily Returns')
    axes[0].set_xlabel('Day of the Month')
    axes[0].set_ylabel('Average Return')
    axes[0].set_xticks(range(1, 32))
    axes[0].set_xticklabels([str(i) for i in range(1, 32)], rotation=45)
    axes[0].grid(True)

    # Plot lunar day patterns with error bars
    axes[1].bar(lunar_day_stats.index, lunar_day_stats['mean'], yerr=lunar_day_stats['sem'], capsize=5)
    axes[1].set_title('Average Returns Grouped by Lunar Days with Standard Error')
    axes[1].set_xlabel('Lunar Day')
    axes[1].set_ylabel('Average Return')
    axes[1].set_xticks(range(1, 31))
    axes[1].set_xticklabels([str(i) for i in range(1, 31)], rotation=45)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'decomposition': decomposition,
        'whole_monthly_patterns': whole_monthly_patterns,
        'monthly_patterns': monthly_patterns,
        'weekly_patterns': weekly_patterns,
        'yearly_weekly_patterns': yearly_weekly_patterns,
        'daily_patterns': daily_patterns,
        'lunar_day_stats': lunar_day_stats
    }


# Example usage
result = analyze_seasonality(stock_symbol='^GSPC', start_date='1924-01-01', end_date='2024-07-01')
print(result)
