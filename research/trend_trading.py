"""
Purpose: to catch the signal of bullish and bearish
Result: back testing and performance evaluation
"""


import pandas as pd
import numpy as np
import yfinance as yf
import talib
import mplfinance as mpf


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


def trade(df):
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
            print(f'buy at {current_price:.2f}')
        elif df['short_signal'][i] and available_stakes < max_stakes:
            # additional conditions for short
            if current_price < base or current_price < peak * 0.95:
                available_stakes += 1
                positions.append(-1)
                print(f'sell at {current_price:.2f}')
            else:
                positions.append(0)
        else:

            positions.append(0)

    return positions


def analyze(ticker, stock_data):
    # Load data
    df = stock_data
    _, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # ta features
    # momentum
    df['ROC'] = talib.ROC(close, timeperiod=21)
    df['RSI'] = talib.RSI(close, timeperiod=21)  # 14, when volatility window=21
    df['water'] = close.diff()
    df['water_delta'] = close.diff().diff()
    # volume
    # volatility
    df['BBU'], _, df['BBL'] = talib.BBANDS(close, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
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
    print(f'\n---- {ticker} ---- ')

    df['position'] = trade(df)

    # Calculate cumulative profit and loss
    df['pnl'] = df['position'] * df['Close'] * (-1)
    df['cumulative_pnl'] = df['pnl'].cumsum()

    # profit per trade
    unsold = df['position'].sum()
    sold = len(df.loc[df['position'] == -1])
    profit = (df['cumulative_pnl'].iloc[-1] + unsold * df['Close'].iloc[-1]) / (sold + unsold) if (sold + unsold) else 0
    # reference point of profit/loss if there was a trade [0, -1]
    reference = df['Close'].iloc[-1] - df['Close'].iloc[0]

    if profit:
        print(f'[{profit:.2f} vs {reference:.2f}]')
        df.to_csv(f'./stock_data/{ticker}.csv')

    plot_trades(ticker, df)

    return [ticker, df['Close'].iloc[-1], df['Close'].max(), df['Close'].min(), profit, reference]


def back_testing(tickers, frequency):

    print(f'Processing {len(tickers)} tickers with frequency {frequency}...')

    data = yf.download(tickers, period=f"{frequency[0]}", interval=f'{frequency[1]}', progress=True, group_by='Ticker')

    # building reports
    df = pd.DataFrame(columns=['Ticker', 'Price', 'Max', 'Min', 'Profit', 'Reference'])

    for i in range(len(tickers)):
        ticker = tickers[i]
        stock_data = data[ticker] if ticker in data.columns else data
        bt = analyze(ticker, stock_data.copy())
        df.loc[i] = bt

    # Performance review - only look into traded tickers (with signals)
    trades = df.loc[df['Profit'] != 0].copy()
    review(trades)


if __name__ == '__main__':
    tickers = ['TSLA']  # 'NVDA', 'NFLX'
    frequency = ['7d', '1m']  # 7 days period, 1-hour interval
    back_testing(tickers, frequency)
