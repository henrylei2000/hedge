from yahoo_fin import stock_info as si
import pandas as pd
import pandas_datareader.data as pdr
import datetime


def save_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv("S&P500.csv", columns=['Symbol'])


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
        sanitized_symbol = symbol  # .replace(".", "-")
        tickers.append(sanitized_symbol)
        # if unique_id > 10:
        #     break
    return tickers


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
    try:
        df = si.get_data(ticker, start_date=a_week_ago, end_date=yesterday)
        change_rate = (df.iloc[4, 4] - df.iloc[0, 4]) / df.iloc[0, 4]
    except IndexError as e:
        change_rate = 0.0
    return change_rate * 100


def sort_change_rate():
    tickers = get_sp500_tickers()  # ["BRK.B"]
    rates = []
    for t in tickers:
        sanitized_symbol = t.replace(".", "-")
        change_rate = get_change_rate(sanitized_symbol)
        if change_rate >= 0:
            print(f" {change_rate}  {t}")
        else:
            print(f" {change_rate}  {t}")
        rates.append(change_rate)

    df = pd.DataFrame(list(zip(tickers, rates)), columns=["Ticker", "Change"])

    sorted_df = df.sort_values(by=['Change'], ascending=False)

    print(sorted_df.to_string())


if __name__ == '__main__':
    sort_change_rate()
