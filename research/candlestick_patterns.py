import pandas as pd
import numpy as np
import yfinance as yf
import talib
import mplfinance as mpf
from scraper.market_watch import fund_estimate

def get_candidate_tickers():
    file = "candidate_tickers.csv"
    df = pd.read_csv(file)
    return df['Ticker']


def plot_trades(df):
    def tradings(df):
        bought, sold = [], []
        for _, row in df.iterrows():
            if row['position'] == 1:
                bought.append(row['Close'])
                sold.append(np.nan)
            elif row['position'] == -1:
                bought.append(np.nan)
                sold.append(row['Close'])
            else:
                bought.append(np.nan)
                sold.append(np.nan)
        return bought, sold

    buy_points, sell_points = tradings(df)

    bbs = df[['BBL', 'BBU']]  # DataFrame with two columns
    apds = [mpf.make_addplot(bbs.iloc[:-1, :], color='r'),
            mpf.make_addplot(df['RSI'][:-1], linestyle='dotted', color='grey')]
    if np.isfinite(buy_points).any():
        apds.append(mpf.make_addplot(buy_points[:-1], type='scatter', markersize=200, marker='^'))
    if np.isfinite(sell_points).any():
        apds.append(mpf.make_addplot(sell_points[:-1], type='scatter', markersize=200, marker='v'))
    mpf.plot(df.iloc[:-1, :], type='candle', volume=True, addplot=apds, show_nontrading=False, figsize=(20, 12))


def review(trades):
    profit, reference, price = trades['Profit'], trades['Reference'], trades['Price']
    positive = len(trades[profit > reference])
    negative = len(trades[profit < reference])
    print(f"[Performance] {len(trades)} traded, {positive} won, {negative} lost, P/L: {profit.sum():.2f}")
    if len(trades) > 3:
        trades['Performance'] = (profit - reference) / price
        best_trades = trades.sort_values('Performance', ascending=False)
        print('------------ TOP 3 --------------')
        print(best_trades[['Ticker', 'Price', 'Profit', 'Reference', 'Performance']].head(3))
        print('------------ BOTTOM 3 --------------')
        print(best_trades[['Ticker', 'Price', 'Profit', 'Reference', 'Performance']].tail(3))


def recommend(rec):
    if len(rec):
        print('------------ RECOMMENDATION --------------')
        buying = rec[rec['Recommendation'] == 'BUY']
        selling = rec[rec['Recommendation'] == 'SELL']
        if len(buying):
            estimates = fund_estimate(buying['Ticker'])  # reference analyst estimate from marketwatch.com
            report = pd.merge(buying, estimates, on='Ticker')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.float_format', '{:.2f}'.format)
            print(report[['Ticker', 'Price', 'Max', 'Min', 'Target', 'Consensus', 'Category']])
            report.to_csv(f'./reports/report-{str(pd.Timestamp.now()):.16s}.csv')
        if len(selling):
            print(selling)
    else:
        print('No recommendations, have a tea \U0001F375')


def tech_analyze(ticker, stock_data, window=20, verbose=False):
    # Load data
    df = stock_data
    _, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # ta features
    # momentum
    df['RSI'] = talib.RSI(close, timeperiod=window)
    # volume
    # volatility
    df['BBU'], _, df['BBL'] = talib.BBANDS(close, timeperiod=window, nbdevup=2, nbdevdn=2, matype=0)
    # trend

    # Define trading signals
    rsi_buy = df['RSI'] < 35
    rsi_sell = df['RSI'] > 70
    bollinger_buy = (close < df['BBL'])
    bollinger_sell = (close > df['BBU'])

    df['long_signal'] = bollinger_buy & rsi_buy
    df['short_signal'] = bollinger_sell & rsi_sell

    # Execute trades
    if verbose:
        print(f'\n---- {ticker} [window={window}] ---- ')

    position = 0
    positions = []
    max_stakes = 10
    for i in range(len(df)):
        if df['long_signal'][i] and position < max_stakes:
            position += 1
            positions.append(1)
            if verbose:
                print(f'buy at {df["Close"][i]:.2f}')
        elif df['short_signal'][i] and position > 0:
            position -= 1
            positions.append(-1)
            if verbose:
                print(f'sell at {df["Close"][i]:.2f}')
        else:
            positions.append(0)
    df['position'] = positions

    # Calculate cumulative profit and loss
    df['pnl'] = df['position'] * df['Close'] * (-1)
    df['cumulative_pnl'] = df['pnl'].cumsum()

    # profit per trade
    unsold = df['position'].sum()
    sold = len(df.loc[df['position'] == -1])
    profit = (df['cumulative_pnl'].iloc[-1] + unsold * df['Close'].iloc[-1]) / (sold + unsold) if (sold + unsold) else 0
    # reference point of profit/loss if there was a trade [0, -1]
    reference = df['Close'].iloc[-1] - df['Close'].iloc[0]

    if verbose and profit:
        print(f'[{profit:.2f} vs {reference:.2f}]')
        df.to_csv(f'./stock_data/{ticker}.csv')

    recommendation = np.nan
    if df['long_signal'][-1] or df['long_signal'][-2]:  # Last TWO signals for a buy recommendation
        if verbose:
            print(f'-------------------------------------------------------------- BUY {ticker}')
        recommendation = 'BUY'
    if df['short_signal'][-1]:
        if verbose:
            print(f'-------------------------------------------------------------- SELL {ticker}')
        recommendation = 'SELL'

    if verbose:
        plot_trades(df)

    return [ticker, df['Close'].iloc[-1], df['Close'].max(), df['Close'].min(), profit, reference, recommendation]


def back_testing(tickers=None, frequency=['90d', '1d'], recommend=False, window=[18]):
    if len(tickers):  # given tickers
        verbose = True
    else:  # all tickers
        tickers = get_candidate_tickers().tolist()
        verbose = False

    print(f'Processing {len(tickers)} tickers with frequency {frequency}...')

    data = yf.download(tickers, period=f"{frequency[0]}", interval=f'{frequency[1]}', progress=True, group_by='Ticker')

    if len(window) == 1:
        b, e, s = window[0], window[0] + 1, 2
    elif len(window) == 2:
        b, e, s = window[0], window[1], 2
    elif len(window) == 3:
        b, e, s = window[0], window[1], window[2]

    for time_window in range(b, e, s):
        print(f'\U0001F375 time_window = {time_window}')
        # building reports
        df = pd.DataFrame(columns=['Ticker', 'Price', 'Max', 'Min', 'Profit', 'Reference', 'Recommendation'])

        for i in range(len(tickers)):
            ticker = tickers[i]
            stock_data = data[ticker] if ticker in data.columns else data
            bt = tech_analyze(ticker, stock_data.copy(), time_window, verbose)
            df.loc[i] = bt

        # Performance review - only look into traded tickers (with signals)
        trades = df.loc[df['Profit'] != 0].copy()
        review(trades)

        # Recommendations on buy and sell - based on the signal in the last (or last two) interval(s)
        if recommend:
            rec = df.loc[~df['Recommendation'].isnull()]
            recommend(rec)


if __name__ == '__main__':
    tickers = ['TQQQ']
    tickers = []  # to have a FULL scan

    low_frequency = ['90d', '1d']  # 90 days period, 1 day interval
    high_frequency = ['7d', '30m']  # 7 days period, 5 minutes interval

    recommend = False
    window = [18, 25, 2]  # time_window beginning, end, and step

    back_testing(tickers, high_frequency, recommend, window)
