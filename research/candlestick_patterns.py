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


def plot_trades(ticker, df):
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
    apds = [mpf.make_addplot(bbs, color='r'),
            mpf.make_addplot(df['RSI'], linestyle='dotted', color='grey', secondary_y=True)]
    if np.isfinite(buy_points).any():
        apds.append(mpf.make_addplot(buy_points, type='scatter', markersize=100, marker='^'))
    if np.isfinite(sell_points).any():
        apds.append(mpf.make_addplot(sell_points, type='scatter', markersize=100, marker='v'))
    mpf.plot(df, type='candle',
             title=f'{ticker}, {df["Close"].min():.2f} - {df["Close"][-1]:.2f} - {df["Close"].max():.2f}',
             volume=True, addplot=apds, show_nontrading=False, figsize=(20, 12))


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


def recommend_trade(rec):
    if len(rec):
        print('------------ RECOMMENDATION --------------')
        buying = rec[rec['Recommendation'] == 'BUY']
        selling = rec[rec['Recommendation'] == 'SELL']
        if len(buying):
            estimates = fund_estimate(buying['Ticker'])  # reference analyst estimate from marketwatch.com
            report = pd.merge(buying, estimates, on='Ticker')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.float_format', '{:.2f}'.format)
            print(report[['Ticker', 'Price', 'Max', 'Min', 'Potential', 'Consensus', 'Category']])
            report.to_csv(f'./reports/report-{str(pd.Timestamp.now()):.16s}.csv')
        if len(selling):
            print(selling)
    else:
        print('No recommendations, have a tea \U0001F375')


def tech_analyze(ticker, stock_data, window=21, verbose=False):
    # Load data
    df = stock_data
    _, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # ta features
    # momentum
    df['RSI'] = talib.RSI(close, timeperiod=window-7)  # 14, when volatility window=21
    df['water'] = close.diff()
    df['water_delta'] = close.diff().diff()
    # volume
    # volatility
    df['BBU'], _, df['BBL'] = talib.BBANDS(close, timeperiod=window, nbdevup=2, nbdevdn=2, matype=0)
    # trend

    # Define trading signals - to be tuned according to the nature of candidates
    rsi_buy = df['RSI'] < 30
    rsi_sell = df['RSI'] > 70

    bollinger_buy = (close < df['BBL'])
    bollinger_sell = (close > df['BBU'])
    diff_buy = (df['water'] / close > 0.02) | (df['water_delta'] / close > 0.003)
    diff_sell = (df['water'] / close < -0.01) | (df['water_delta'] / close < 0)

    df['long_signal'] = diff_buy
    df['short_signal'] = diff_sell


    # Execute trades
    if verbose:
        print(f'\n---- {ticker} [window={window}] ---- ')

    hold = False
    max_stakes = 1
    available_stakes = max_stakes
    positions = []
    peak, base = 0, 0

    for i in range(len(df)):
        current_price = df["Close"][i]
        peak = max(peak, current_price)
        if df['long_signal'][i] and available_stakes > 0:
            available_stakes -= 1
            base = current_price
            peak = base  # reset peak
            positions.append(1)
            if verbose:
                print(f'buy at {current_price:.2f}')
        elif df['short_signal'][i] and available_stakes < max_stakes:
            # additional conditions for short
            if current_price < base or current_price < peak * 0.95:
                available_stakes += 1
                positions.append(-1)
                if verbose:
                    print(f'sell at {current_price:.2f}')
            else:
                positions.append(0)
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
        plot_trades(ticker, df)

    return [ticker, df['Close'].iloc[-1], df['Close'].max(), df['Close'].min(), profit, reference, recommendation]


def back_testing(tickers=None, frequency=['3mo', '1d'], recommend=False, window=[18]):
    if len(tickers):  # given tickers
        verbose = True
    else:  # all tickers
        tickers = get_candidate_tickers().tolist()
        verbose = False

    print(f'Processing {len(tickers)} tickers with frequency {frequency}...')

    data = yf.download(tickers, period=f"{frequency[0]}", interval=f'{frequency[1]}', progress=True, group_by='Ticker')
    # beginning, end, step for time window parameters
    if len(window) == 1:
        b, e, s = window[0], window[0] + 1, 2
    elif len(window) == 2:
        b, e, s = window[0], window[1], 2
    elif len(window) == 3:
        b, e, s = window[0], window[1], window[2]
    else:
        b, e, s = 21, 22, 1

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
            recommend_trade(rec)


def macd():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load historical price data
    data = pd.read_csv("./stock_data/CFG.csv")
    data["Date"] = pd.to_datetime(data["Datetime"])
    data.set_index("Date", inplace=True)

    # Calculate MACD indicator
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Plot MACD and signal lines
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, macd, label="MACD")
    plt.plot(data.index, signal, label="Signal")
    plt.legend()

    # Generate trading signals based on MACD and signal lines
    data["MACD"] = macd
    data["Signal"] = signal
    data["Signal_Crossover"] = np.where(data["MACD"] > data["Signal"], 1, -1)

    # Plot price data with trading signals
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Price")
    plt.scatter(data[data["Signal_Crossover"] == 1].index, data[data["Signal_Crossover"] == 1]["Close"], label="Buy",
                marker="^", color="green")
    plt.scatter(data[data["Signal_Crossover"] == -1].index, data[data["Signal_Crossover"] == -1]["Close"], label="Sell",
                marker="v", color="red")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tickers = ['TSLA'] # 'NVDA', 'NFLX'
    # tickers = []  # to have a FULL scan, remember to turn on the recommend flag below
    recommend = True

    low_frequency = ['1y', '1d']  # 90 days period, 1-day interval
    high_frequency = ['7d', '1m']  # 7 days period, 1-hour interval

    window = [20, 21, 2]  # time_window beginning, end, and step

    back_testing(tickers, high_frequency, recommend, window)