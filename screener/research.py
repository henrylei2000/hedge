from yahoo_fin import stock_info as si
import pandas as pd
import pandas_datareader.data as pdr
import datetime


def save_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv("S&P500.csv", columns=['Symbol', 'GICS Sector'])


def save_nasdaq100_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
    df = table[4]
    try:
        df.to_csv("Nasdaq100.csv", columns=['Ticker', 'GICS Sector'])
    except Exception as ex:
        print(ex)
        pass


def get_nasdaq100_tickers():
    print("updating nasdaq 100 tickers...")
    save_nasdaq100_tickers()
    tickers = []
    df = pd.read_csv("Nasdaq100.csv")
    for i, row in df.iterrows():
        unique_id = i
        symbol = row['Ticker']
        sector = row['GICS Sector']
        sanitized_symbol = symbol.replace(".", "-")
        tickers.append((sanitized_symbol, sector))

    return tickers  # tuples of (ticker, sector)


def read_table():
    income_statement = si.get_income_statement("tsla")
    print(income_statement)

    quote_table = si.get_quote_table("tsla", dict_result=False)
    print(quote_table)


def read_asset(ticker):
    table = pd.read_html(f'https://www.marketwatch.com/investing/stock/{ticker}/financials/balance-sheet')
    try:
        t = table[4] .drop(columns=["5-year trend"])
        for i in range(len(t)):
            if t.iloc[i, 0] == "Total Assets  Total Assets":
                print(ticker + ', ' + t.iloc[i, 5])
                return t.iloc[i, 5]
    except IndexError as e:
        pass

    return "0B"


def read_cap(ticker):
    sanitized_symbol = ticker.replace(".", "-")
    cap = pdr.get_quote_yahoo(sanitized_symbol)['marketCap']
    return cap.values[0]


def get_sp500_tickers():
    print("updating s&p500 tickers...")
    save_sp500_tickers()
    tickers = []
    df = pd.read_csv("S&P500.csv")
    for i, row in df.iterrows():
        unique_id = i
        symbol = row['Symbol']
        sector = row['GICS Sector']
        sanitized_symbol = symbol  # .replace(".", "-")
        tickers.append((sanitized_symbol, sector))
        # if unique_id > 10:
        #     break
    return tickers  # tuples of (ticker, sector)


def sort_market_cap():
    tickers = get_sp500_tickers()  # ["BRK.B"]
    assets = []
    caps = []
    for t in tickers:
        print(f"Processing {t} ...")
        string_value = read_asset(t)
        assets.append(float(string_value[0:-1]))
        caps.append(read_cap(t))

    df = pd.DataFrame(list(zip(tickers, assets, caps)), columns=["Ticker", "Asset", "Cap"])

    sorted_df = df.sort_values(by=['Cap'], ascending=False)

    print(sorted_df)


def get_change_rate(ticker):
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    a_week_ago = yesterday - datetime.timedelta(days=7)
    a_month_ago = yesterday - datetime.timedelta(days=30)
    year = datetime.date.today().year
    year_beginning = datetime.datetime(year, 1, 1)
    a_year_ago = yesterday - datetime.timedelta(days=365)
    try:
        df = si.get_data(ticker, start_date=a_week_ago, end_date=yesterday)
        change_rate = (df.iloc[-1, 4] - df.iloc[0, 4]) / df.iloc[0, 4]  # iloc[-1]: last row
    except IndexError as e:
        change_rate = 0.0
    return change_rate * 100


def sort_change_rate():
    #tickers = get_sp500_tickers()  # ["BRK.B"]
    tickers = get_nasdaq100_tickers()
    rates = []
    selected_tickers = []
    selected_sectors = []
    selected_rates = []
    for t, s in tickers:
        sanitized_symbol = t.replace(".", "-")
        change_rate = get_change_rate(sanitized_symbol)
        selected_rates.append(change_rate)
        selected_tickers.append(t)
        selected_sectors.append(s)

        print(f"{change_rate}  {t}")
        rates.append(change_rate)

    df = pd.DataFrame(list(zip(selected_tickers, selected_sectors, selected_rates)), columns=["Ticker", "GICS Sector", "Change"])

    sorted_df = df.sort_values(by=['Change'], ascending=False)

    print(sorted_df.head(10))
    print(sorted_df.tail(10).iloc[::-1])


if __name__ == '__main__':
    sort_change_rate()
