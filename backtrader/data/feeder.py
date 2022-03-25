from pandas_datareader import data as pdr
import yfinance as yf
import datetime


def stock_to_csv(ticker,
                 start_date=datetime.datetime.now() - datetime.timedelta(days=366),
                 end_date=datetime.datetime.now() - datetime.timedelta(days=1),
                 source="import"):

    # fast downloader
    yf.pdr_override()

    # Download historical data as CSV for each stock (makes the process faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)

    try:
        if source == "main":
            df.to_csv(f'{ticker}.csv')
        else:
            df.to_csv(f'data/{ticker}.csv')
    except Exception:
        print(f'Ticker {ticker}: No data or delisted.')


def main():
    ticker = "NIO"
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    one_year_ago = yesterday - datetime.timedelta(days=365)
    stock_to_csv(ticker, one_year_ago, yesterday, "main")


if __name__ == "__main__":
    main()
