import yfinance as yf
import pandas as pd
import mplfinance as mpf
import talib # Import TA-Lib
import matplotlib.pyplot as plt
import numpy as np

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



def ta_bbands(ticker, plot=False):
    import pandas as pd
    import ta

    # Load data
    df = yf.download(ticker, period="5d", interval='5m', progress=False)

    # Calculate EMA
    df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)

    # Calculate RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['vwap'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

    # Calculate Bollinger Bands
    df['BBL'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BBM'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
    df['BBU'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)

    # Define trading signals
    ema_buy = (df['Close'] > df['EMA5']) & (df['EMA5'] > df['EMA10'])
    ema_sell = (df['Close'] < df['EMA5']) & (df['EMA5'] < df['EMA10'])
    rsi_buy = (df['RSI'] < 30) | (df['RSI'].shift(2) - df['RSI'] > 10)
    rsi_sell = df['RSI'] > 72
    bbands_buy = df['Close'] < df['BBL']
    bbands_sell = df['Close'] > df['BBU']
    vwap_buy = (df['Close'] > df['vwap']) & (df['Close'].shift(1) <= df['vwap'].shift(1))
    vwap_sell = (df['Close'] < df['vwap']) & (df['Close'].shift(1) >= df['vwap'].shift(1))
    df['long_signal'] = bbands_buy & rsi_buy
    df['short_signal'] = bbands_sell & rsi_sell

    # Execute trades
    position = 0
    positions = []
    for i in range(len(df)):
        if df['long_signal'][i] and position < 5:
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

        buy_signals = buy(df)
        sell_signals = sell(df)

        bbs = df[['BBL', 'BBU']]  # DataFrame with two columns

        apds = [mpf.make_addplot(bbs.iloc[:-1, :], color='red'), mpf.make_addplot(df['RSI'][:-1])]
        if np.isfinite(buy_signals).any():
            apds.append(mpf.make_addplot(buy_signals[:-1], type='scatter', markersize=50, marker='^'))
        if np.isfinite(sell_signals).any():
            apds.append(mpf.make_addplot(sell_signals[:-1], type='scatter', markersize=50, marker='v'))
        mpf.plot(df.iloc[:-1, :], type='candle', volume=True, addplot=apds)
        # mpf.plot(df.tail(100), type='candle', volume=True, mav=(10, 20), show_nontrading=False)

    return [ticker, profit, base]

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


def animate_stock():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import yfinance as yf

    # Download stock data
    stock = yf.download('AMZN', period='1mo')

    # Remove rows with NaN or infinite values
    stock = stock.dropna()

    # Define figure and axis
    fig, ax = plt.subplots()

    # Initialize line plot
    line, = ax.plot(stock.index, stock['Close'])

    # Define animation function
    def animate(i):
        # Update data
        line.set_data(stock.index[:i], stock['Close'][:i])
        # Update x-axis limits
        ax.set_xlim(stock.index[0], stock.index[-1])
        # Update y-axis limits
        ax.set_ylim(min(stock['Close']), max(stock['Close']))
        # Set title
        ax.set_title('AMZN Stock Price')
        # Set x-label
        ax.set_xlabel('Date')
        # Set y-label
        ax.set_ylabel('Price')
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Animate plot
    ani = FuncAnimation(fig, animate, frames=len(stock.index), interval=150)

    # Show plot
    plt.show()


if __name__ == '__main__':
    # backtesting()
    ta_bbands('ADBE', plot=True)
