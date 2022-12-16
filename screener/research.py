from yahoo_fin import stock_info as si
import pandas as pd
import pandas_ta as ta
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


def get_sector_tickers():
    print("updating spdr sector tickers...")
    tickers = [('XLY', 'Consumer Discretionary'), ('XLK', 'Technology'),
               ('XLC', 'Communication Services'), ('XLI', 'Industrials'), ('XLF', 'Financial Services'), ('XLB', 'Materials'),
               ('XLE', 'Energy'), ('XLV', 'Health Care'), ('XLRE', 'Real Estate'), ('XLU', 'Utilities'), ('XLP', 'Consumer Staples')]

    return tickers  # list of tuples


def get_index_tickers():
    print("updating index tickers...")
    tickers = [('SPY', 'S&P 500'), ('QQQ', 'NASDAQ 100'), ('DIA', 'Dow-Jones')]

    return tickers  # list of tuples


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


def get_change_rate(ticker, start):
    today = datetime.date.today()
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

    if start == "a_week_ago":
        start_date = datetime.date(today.year, today.month, today.day - 7)
    elif start == "a_month_ago":
        start_date = datetime.date(today.year, today.month - 1, today.day)
    elif start == "two_months_ago":
        start_date = datetime.date(today.year, today.month - 2, today.day)
    elif start == "three_months_ago":
        start_date = datetime.date(today.year, today.month - 3, today.day)
    elif start == "six_months_ago":
        start_date = datetime.date(today.year, today.month - 6, today.day)
    elif start == "ytd":
        start_date = datetime.date(today.year, 1, 1)
    elif start == "a_year_ago":
        start_date = datetime.date(today.year - 1, today.month, today.day)
    elif start == "two_years_ago":
        start_date = datetime.date(today.year - 2, today.month, today.day)
    elif start == "five_years_ago":
        start_date = datetime.date(today.year - 5, today.month, today.day)
    elif start == "ten_years_ago":
        start_date = datetime.date(today.year - 10, today.month, today.day)
    elif start == "twenty_years_ago":
        start_date = datetime.date(today.year - 20, today.month, today.day)
    elif start == "thirty_years_ago":
        start_date = datetime.date(today.year - 30, today.month, today.day)
    else:
        print("Invalid time frame, please try again.")
        return 0

    try:
        df = si.get_data(ticker, start_date, end_date=yesterday)
        change_rate = (df.iloc[-1, 4] - df.iloc[0, 4]) / df.iloc[0, 4]  # iloc[-1]: last row
    except IndexError as e:
        change_rate = 0.0
    return change_rate * 100


def sort_change_rate(market="indices", start="a_week_ago"):
    if market == "s&p500":
        tickers = get_sp500_tickers()  # ["BRK.B"]
    elif market == "nasdaq100":
        tickers = get_nasdaq100_tickers()
    elif market == "sectors":
        tickers = get_sector_tickers()
    else:
        tickers = get_index_tickers()

    rates = []
    symbols = []
    sectors = []
    for t, s in tickers:
        sanitized_symbol = t.replace(".", "-")
        change_rate = get_change_rate(sanitized_symbol, start)
        rates.append(change_rate)
        symbols.append(t)
        sectors.append(s)

        print(f"{change_rate}  {t}")

    df = pd.DataFrame(list(zip(symbols, sectors, rates)), columns=["Ticker", "Sector", "Change"])

    sorted_df = df.sort_values(by=['Change'], ascending=False)

    print("\n\n****************************\n\n")
    print(f"{market} change rate since {start}\n\n")
    print(sorted_df.head(13))
    # print(sorted_df.tail(10).iloc[::-1])


if __name__ == '__main__':
    sort_change_rate("indices", "a_year_ago")
