import yfinance as yf
import pandas as pd
import mplfinance as mpf
import talib # Import TA-Lib
import matplotlib.pyplot as plt
import numpy as np
import ta  # https://github.com/bukosabino/ta/

def visualize():
    # Set the ticker symbol and time interval
    ticker = "AMZN"
    interval = "5m"

    # Set the date range
    start_date = pd.Timestamp.now() - pd.DateOffset(days=59)
    end_date = pd.Timestamp.now()

    # Retrieve the data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # plot the candlesticks using mplfinance
    mpf.plot(data.tail(100), type='candle', volume=True, mav=(10, 20), show_nontrading=False)


def sma():
    # Load AMZN stock price data
    df = yf.download("SIVB", period="3mo")

    # Calculate moving averages
    df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)

    # Create a new column to indicate buy/sell signals
    df['Signal'] = 0
    df.loc[df['SMA20'] > df['SMA50'] + 1, 'Signal'] = 1
    df.loc[df['SMA20'] < df['SMA50'] - 1, 'Signal'] = -1

    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()

    # Calculate strategy returns
    df['Strategy'] = df['Signal'].shift(1) * df['Return']

    # Calculate cumulative returns
    df['Cumulative_Returns'] = (1 + df['Strategy']).cumprod() - 1

    print('1year, daily, sma', df['Cumulative_Returns'].iloc[-1])
    # Plot cumulative returns
    # df['Cumulative_Returns'].plot()
    # plt.show()


def candlesticks():
    import pandas_ta as ta

    # Load stock data
    import pandas_ta as ta

    df = yf.download("AMZN", period="1mo", interval="5m")
    # Access the index column (first column)
    index_col = df.index

    # Rename the index column
    df = df.rename_axis('Datetime').reset_index()

    # Now the index column is named 'Date'
    print(df.keys())
    print(index_col)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)

    # Calculate Engulfing pattern
    df.ta.cdl_pattern(name="hammer", append=True)
    df['signal'] = 0
    df.loc[(df['CDL_HAMMER'] == -100) & (df['RSI_14'] > 70), 'signal'] = -1
    df.loc[(df['CDL_HAMMER'] == -100) & (df['RSI_14'] < 30), 'signal'] = 1

    # Calculate returns and cumulative returns
    df['return'] = df['Close'].pct_change() * df['signal'].shift(1)
    df['cum_return'] = (1 + df['return']).cumprod() - 1

    # Print cumulative return
    print('1month, 5m, hammer', df['cum_return'].iloc[-1])
    # Print dataframe with Engulfing pattern values
    df.dropna(inplace=True, how='any')
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.layout = dict(xaxis=dict(type="category"))
    fig.show()


def get_tmt_tickers():
    file = "tmt_nasdaq100.csv"
    df = pd.read_csv(file)
    return df['Ticker']


def backtesting():
    import numpy as np
    tickers = get_tmt_tickers()
    df = pd.DataFrame(columns=['Ticker', 'Profit', 'Base', 'Good'])
    for t in tickers:
        bt = ta_bbands(t)
        if bt[1] > bt[2]:
            bt.append(1)
        else:
            bt.append('')
        df.loc[len(df)] = bt
    print(df)
    print(len(df), '-----------', df['Good'].value_counts()[1])

    print(df['Profit'].sum())


def ta_bbands(ticker, plot=False):
    # Load data
    df = yf.download(ticker, period="5d", interval='5m', progress=False)

    open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    
    # ta features
    # momentum
    df['RSI'] = ta.momentum.rsi(close, window=14, fillna=False)
    df['TSI'] = ta.momentum.tsi(close, window_slow=25, window_fast=13, fillna=False)
    df['UO'] = ta.momentum.ultimate_oscillator(high, low, close, window1=7, window2=14, window3=28,
                                               weight1=4.0, weight2=2.0, weight3=1.0, fillna=False)
    df['STOCH'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3, fillna=False)
    df['STOCH_SIGNAL'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    df['STOCH_RSI'] = ta.momentum.stochrsi(close, window=14, smooth1=3, smooth2=3, fillna=False)
    df['KAMA'] = ta.momentum.kama(close, window=10, pow1=2, pow2=30, fillna=False)
    df['ROC'] = ta.momentum.roc(close, window=12, fillna=False)
    df['AO'] = ta.momentum.awesome_oscillator(high, low, window1=5, window2=34, fillna=False)
    df['WILLIAMS_R'] = ta.momentum.williams_r(high, low, close, lbp=14, fillna=False)
    df['PPO'] = ta.momentum.ppo(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['PVO'] = ta.momentum.pvo(volume, window_slow=26, window_fast=12, window_sign=9, fillna=False)

    # volume
    df['ADI'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=False)
    df['OBV'] = ta.volume.on_balance_volume(close, volume, fillna=False)
    df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=False)
    df['FI'] = ta.volume.force_index(close, volume, window=13, fillna=False)
    df['EoM'] = ta.volume.ease_of_movement(high, low, volume, window=14, fillna=False)
    df['SMA_EoM'] = ta.volume.sma_ease_of_movement(high, low, volume, window=14, fillna=False)
    df['VPT'] = ta.volume.volume_price_trend(close, volume, fillna=False)
    df['NVI'] = ta.volume.negative_volume_index(close, volume, fillna=False)
    df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14, fillna=False)
    df['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)

    # volatility
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14, fillna=False)
    df['BBM'] = ta.volatility.bollinger_mavg(close, window=20, fillna=False)
    df['BBU'] = ta.volatility.bollinger_hband(close, window=20, window_dev=2, fillna=False)
    df['BBL'] = ta.volatility.bollinger_lband(close, window=20, window_dev=2, fillna=False)
    df['BBW'] = ta.volatility.bollinger_wband(close, window=20, window_dev=2, fillna=False)
    df['BBP'] = ta.volatility.bollinger_pband(close, window=20, window_dev=2, fillna=False)
    # Keltner ... skipped
    # Donchian
    df['DCM'] = ta.volatility.donchian_channel_mband(high, low, close, window=10, offset=0, fillna=False)
    df['DCU'] = ta.volatility.donchian_channel_hband(high, low, close, window=20, offset=0, fillna=False)
    df['DCL'] = ta.volatility.donchian_channel_lband(high, low, close, window=20, offset=0, fillna=False)
    df['DCW'] = ta.volatility.donchian_channel_wband(high, low, close, window=10, offset=0, fillna=False)
    df['DCP'] = ta.volatility.donchian_channel_pband(high, low, close, window=10, offset=0, fillna=False)

    # trend
    df['EMA5'] = ta.trend.ema_indicator(close, window=5)
    df['EMA10'] = ta.trend.ema_indicator(close, window=10)
    df['WMA10'] = ta.trend.wma_indicator(close, window=10, fillna=False)
    df['MACD'] = ta.trend.macd(close, window_slow=26, window_fast=12, fillna=False)
    df['MACD_SIGNAL'] = ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['MACD_DIFF'] = ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    # df['ADX'] = ta.trend.adx(high, low, close, window=14, fillna=False)
    df['CCI'] = ta.trend.cci(high, low, close, window=20, constant=0.015, fillna=False)
    df['DPO'] = ta.trend.dpo(close, window=20, fillna=False)
    df['STC'] = ta.trend.stc(close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=False)
    df['PSAR_UP'] = ta.trend.psar_up(high, low, close, step=0.02, max_step=0.20, fillna=False)
    df['PSAR_DOWN'] = ta.trend.psar_down(high, low, close, step=0.02, max_step=0.20, fillna=False)

    # others

    # Define trading signals
    ema_buy = (close > df['EMA5']) & (df['EMA5'] > df['EMA10'])
    ema_sell = (close < df['EMA5']) & (df['EMA5'] < df['EMA10'])
    rsi_buy = ((df['RSI'] < 30) | (df['PPO'] < -2))
    rsi_sell = df['RSI'] > 70
    bbands_buy = close < df['BBL']
    bbands_sell = close > df['BBU']
    vwap_buy = (close > df['VWAP']) & (close.shift(1) <= df['VWAP'].shift(1))
    vwap_sell = (close < df['VWAP']) & (close.shift(1) >= df['VWAP'].shift(1))

    df['long_signal'] = bbands_buy & rsi_buy
    df['short_signal'] = bbands_sell & rsi_sell

    # Execute trades
    position = 0
    positions = []
    entry = 1000
    max_stakes = min(int(entry / df['Close'].mean()), 5)
    max_stakes = max(2, max_stakes)
    for i in range(len(df)):
        if df['long_signal'][i] and position < max_stakes:
            position += 1
            positions.append(1)
            print(f'buy at {df["Close"][i]}')
        elif df['short_signal'][i] and position > 0:
            position -= 1
            positions.append(-1)
            print(f'sell at {df["Close"][i]}')
        else:
            positions.append(0)

    # Calculate profit and loss
    df['position'] = positions

    df['pnl'] = df['position'] * df['Close'] * (-1)
    # df['pnl'] = df['position'] * (df['Close'] - df['Close'].shift(1))

    # Calculate cumulative profit and loss
    df['cumulative_pnl'] = df['pnl'].cumsum()

    # print(df)
    df.to_csv(f'./stock_data/{ticker}.csv')

    profit = df['cumulative_pnl'].iloc[-1] + df['position'].sum() * df['Close'][-1]

    # Plot cumulative returns
    base = df['Close'].iloc[-1] - df['Close'].iloc[0]

    print(f'pnl {profit} vs {base}')

    if plot:
        def buy(df):
            signal = []
            for _, row in df.iterrows():
                if row['position'] == 1:
                    signal.append(row['Close'])
                else:
                    signal.append(np.nan)
            return signal

        def sell(df):
            signal = []
            for _, row in df.iterrows():
                if row['position'] == -1:
                    signal.append(row['Close'] * 1)
                else:
                    signal.append(np.nan)
            return signal

        buy_points = buy(df)
        sell_points = sell(df)

        bbs = df[['BBL', 'BBU']]  # DataFrame with two columns

        apds = [mpf.make_addplot(bbs.iloc[:-1, :], color='red'), mpf.make_addplot(df['RSI'][:-1])]
        if np.isfinite(buy_points).any():
            apds.append(mpf.make_addplot(buy_points[:-1], type='scatter', markersize=50, marker='^'))
        if np.isfinite(sell_points).any():
            apds.append(mpf.make_addplot(sell_points[:-1], type='scatter', markersize=50, marker='v'))
        mpf.plot(df.iloc[:-1, :], type='candle', volume=True, addplot=apds, show_nontrading=False)
        # mpf.plot(df.tail(100), type='candle', volume=True, mav=(10, 20), show_nontrading=False)

    return [ticker, profit, base]

if __name__ == '__main__':
    # backtesting()
    ta_bbands('PDD', plot=True)
