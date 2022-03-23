from pandas_datareader import data as pdr
import yfinance as yf
import datetime


def stock_to_csv(ticker, start_date, end_date):

    # fast downloader
    yf.pdr_override()

    # Download historical data as CSV for each stock (makes the process faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)

    try:
        df.to_csv(f'{ticker}.csv')
    except Exception:
        print(f'Ticker {ticker}: No data or delisted.')


def main():
    ticker = "TQQQ"
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    one_year_ago = yesterday - datetime.timedelta(days=365)
    stock_to_csv(ticker, one_year_ago, yesterday)


if __name__ == "__main__":
    main()
