import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def save_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500.csv", columns=['Symbol'])


def save_nasdaq100_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
    df = table[3]
    try:
        df.to_csv('Nasdaq100-Info.csv')
        df.to_csv("Nasdaq100.csv", columns=['Ticker'])
    except Exception as ex:
        pass

def get_sp500_tickers():
    tickers = []
    df = pd.read_csv("S&P500.csv")
    for i, row in df.iterrows():
        unique_id = i
        symbol = row['Symbol']
        sanitized_symbol = symbol.replace(".", "-")
        tickers.append(sanitized_symbol)

    return tickers

def get_nasdaq100_tickers():
    tickers = []
    df = pd.read_csv("Nasdaq100.csv")
    for i, row in df.iterrows():
        unique_id = i
        symbol = row['Ticker']
        sanitized_symbol = symbol.replace(".", "-")
        tickers.append(sanitized_symbol)

    return tickers

def save_data(tickers):
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    a_week_ago = yesterday - datetime.timedelta(days=7)
    a_month_ago = yesterday - datetime.timedelta(days=30)
    three_month_ago = yesterday - datetime.timedelta(days=90)
    a_year_ago = yesterday - datetime.timedelta(days=365)
    for t in tickers:
        print(f"{t}...")
        try:
            df = pdr.get_data_yahoo(t, three_month_ago, yesterday)
            df.to_csv(f"data/{t}.csv")
        except Exception as ex:
            print("Retrying on: ", ex)
            time.sleep(10)
            try:
                df = pdr.get_data_yahoo(t, three_month_ago, yesterday)
                df.to_csv(f"data/{t}.csv")
            except Exception as ex:
                print("Failed on:", ex)


def read_data(ticker):
    df = pd.read_csv(f"data/{ticker}.csv")
    df.set_index('Date', inplace=True)
    df.rename(columns={'Adj Close': ticker}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
    print(df.head())
    return df


# save_sp500_tickers()
def compile_data(tickers=None):
    if tickers is None:
        tickers = get_sp500_tickers()

    save_data(tickers)

    # main_df = pd.DataFrame()
    # for count, ticker in enumerate(tickers):
    #         df = read_data(ticker)
    #         if main_df.empty:
    #             main_df = df
    #         else:
    #             main_df = main_df.join(df, how='outer')
    #
    #         if count % 10 == 0:
    #             print(count)

    main_df = pd.concat([read_data(t).reset_index(drop=True) for t in tickers], axis=1)

    print(main_df.head())

    main_df.to_csv('joined_closes.csv')


def visualize_data():
    df = pd.read_csv('joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence


if __name__ == '__main__':

    #save_nasdaq100_tickers()

    tickers = ["DKS", "ASO", "FL", "BBY", "AEO", "ANF"]
    # tickers = get_nasdaq100_tickers()
    # save_data(tickers)
    compile_data(tickers)
    visualize_data()
    # examples of running:
    # do_ml('XOM')
    # do_ml('AAPL')
    # do_ml('ABT')