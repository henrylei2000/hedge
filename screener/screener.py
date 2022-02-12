from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import os


def screen_market(market, start_date, end_date):
    index_name = ""
    # Variables
    if market == "^GSPC":
        index_name = '^GSPC'  # S&P 500
        tickers = si.tickers_sp500()
    else:
        index_name = 'QQQ'  # Nasdap 100
        tickers = si.tickers_nasdaq()

    tickers = [item.replace(".", "-") for item in tickers]  # Yahoo Finance uses dashes instead of dots

    returns_multiples = []

    # Index Returns
    index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
    index_df['Percent Change'] = index_df['Adj Close'].pct_change()
    index_return = (index_df['Percent Change'] + 1).cumprod()[-1]

    # fast downloader
    yf.pdr_override()

    # Find top 30% performing stocks (relative to the S&P 500)
    for ticker in tickers:
        # Download historical data as CSV for each stock (makes the process faster)
        df = pdr.get_data_yahoo(ticker, start_date, end_date)

        try:
            # Calculating returns relative to the market (returns multiple)
            df['Percent Change'] = df['Adj Close'].pct_change()
            stock_return = (df['Percent Change'] + 1).cumprod()[-1]

            returns_multiple = round((stock_return / index_return), 2)
            returns_multiples.extend([returns_multiple])

            print(f'Ticker: {ticker}; Returns Multiple against {market} index: {returns_multiple}\n')

            df.to_csv(f'{ticker}.csv')

        except Exception:
            print(f'Ticker {ticker}: No data or delisted.')

    # Creating dataframe of only top 30%
    rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
    rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
    rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.70)]
    print(f"rs_df len: {len(rs_df)}")
    # Checking Minervini conditions of top 30% of stocks in given list
    rs_stocks = rs_df['Ticker']
    print(f"rs_stocks len: {len(rs_stocks)}")
    exportList = pd.DataFrame(
        columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High", "Current Close"])
    for stock in rs_stocks:
        try:
            df = pd.read_csv(f'{stock}.csv', index_col=0)

            sma = [50, 150, 200]
            for x in sma:
                df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)

            # Storing required values
            currentClose = df["Adj Close"][-1]
            moving_average_50 = df["SMA_50"][-1]
            moving_average_150 = df["SMA_150"][-1]
            moving_average_200 = df["SMA_200"][-1]
            low_of_52week = round(min(df["Low"][-260:]), 2)
            high_of_52week = round(max(df["High"][-260:]), 2)
            RS_Rating = round(rs_df[rs_df['Ticker'] == stock].RS_Rating.tolist()[0])

            try:
                moving_average_200_20 = df["SMA_200"][-20]
            except Exception:
                moving_average_200_20 = 0

            # Condition 1: Current Price > 150 SMA and > 200 SMA
            condition_1 = currentClose > moving_average_150 > moving_average_200

            # Condition 2: 150 SMA and > 200 SMA
            condition_2 = moving_average_150 > moving_average_200

            # Condition 3: 200 SMA trending up for at least 1 month
            condition_3 = moving_average_200 > moving_average_200_20

            # Condition 4: 50 SMA> 150 SMA and 50 SMA > 200 SMA
            condition_4 = moving_average_50 > moving_average_150 > moving_average_200

            # Condition 5: Current Price > 50 SMA
            condition_5 = currentClose > moving_average_50

            # Condition 6: Current Price is at least 30% above 52 week low
            condition_6 = currentClose >= (1.3 * low_of_52week)

            # Condition 7: Current Price is within 25% of 52 week high
            condition_7 = currentClose >= (.75 * high_of_52week)

            # If all conditions above are true, add stock to exportList
            if (condition_1 and condition_2 and condition_3 and condition_4
                    and condition_5 and condition_6 and condition_7):
                exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating, "50 Day MA": moving_average_50,
                                                "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200,
                                                "52 Week Low": low_of_52week, "52 week High": high_of_52week,
                                                "Current Close": currentClose}, ignore_index=True)
                print(stock + " made the Minervini requirements")

        except Exception as e:
            print(e)
            print(f"Could not gather data on {stock}")

    exportList = exportList.sort_values(by='RS_Rating', ascending=False)
    timestamp = end_date.strftime("%m_%d_%Y")
    exportList.to_csv(f"screen_{timestamp}.csv")
    print('\n', exportList)
    writer = ExcelWriter(f"ScreenOutput_{timestamp}.xlsx")
    exportList.to_excel(writer, "Sheet1")
    writer.save()

    return exportList


def clean_up():
    files = os.listdir("./")
    for file in files:
        if file.endswith(".csv") and not file.startswith("screen") and not file.startswith("transaction"):
            os.remove(file)


def manage_fund(fund_size, fund_date, candidates):
    transactions = pd.DataFrame(
        columns=['Stock', "Share", "Investment", "Current Close"])

    timestamp = fund_date.strftime("%m_%d_%Y")
    previous_fund_date = fund_date - datetime.timedelta(days=30)
    previous_timestamp = previous_fund_date.strftime("%m_%d_%Y")

    if os.path.exists(f'./transaction_{previous_timestamp}.csv'):
        fund_size = 0
        tranx = pd.read_csv(f'./transaction_{previous_timestamp}.csv', index_col=0)
        for index, tran in tranx.iterrows():
            stock = pd.read_csv(f"{tran['Stock']}.csv")
            price = stock.iloc[-1]["Adj Close"]
            fund_size = fund_size + tran['Share'] * price
    else:
        fund_size = fund_size

    # buy
    candidates = candidates.reset_index()  # make sure indexes pair with number of rows
    counter = 0
    if len(candidates.index) > 5:
        max = 5
    else:
        max = len(candidates.index)

    for index, row in candidates.iterrows():
        counter = counter + 1
        if counter > max:
            break
        else:
            investment = fund_size / max
            share = investment / row['Current Close']
            transactions = transactions.append({'Stock': row['Stock'], "Share": share, "Investment": investment,
                                                "Current Close": row['Current Close']}, ignore_index=True)

    transactions.to_csv(f"transaction_{timestamp}.csv")


def main():
    # fund starting point
    index_name = "^GSPC"
    fund_size = 100000
    today = datetime.datetime.now()
    fund_date = today - datetime.timedelta(days=365)

    while fund_date < today:
        screen_start_date = fund_date - datetime.timedelta(days=365)
        screen_end_date = fund_date
        candidates = screen_market(index_name, screen_start_date, screen_end_date)
        manage_fund(fund_size, fund_date, candidates)
        clean_up()
        fund_date = fund_date + datetime.timedelta(days=30)

    """
    regression test Minervini strategy
    
    - transaction_frequency: 1 month, 3 months (default), 6 months, 12 months ...
    
    - start_date = 5 years ago
    - end_date = start_date + 1 year (52 weeks)
    - fund = 100000
    
    - algorithm:
    
        while end_date < now:
            screen_market("Nasdaq", start_date, end_date)
            manage_fund(end_date)
            start_date += 3 months
            end_date += 3 months
    """


if __name__ == "__main__":
    main()