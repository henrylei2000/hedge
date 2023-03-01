import yfinance as yf
import pandas as pd
import mplfinance as mpf
import talib # Import TA-Lib
import matplotlib.pyplot as plt

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

def bbands(ticker):
    import pandas_ta as ta

    # Download AMZN stock price data for the past year
    symbol = 'AMZN'
    # df = yf.download(symbol, period='1y')
    # df = yf.download(symbol, period="1mo", interval="5m")

    # ticker = "AMZN"
    interval = "5m"

    # Set the date range
    end_date = pd.Timestamp.now() - pd.DateOffset(days=0)
    start_date = end_date - pd.DateOffset(days=5)

    # Retrieve the data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    # Calculate Bollinger Bands
    bbs = ta.bbands(df['Close'], length=20, std=2)

    # Create signals based on Bollinger Bands
    df['signal'] = 0
    df.loc[df['Close'] < bbs['BBL_20_2.0'], 'signal'] = 1
    df.loc[df['Close'] > bbs['BBU_20_2.0'], 'signal'] = -1

    tcdf = bbs[['BBL_20_2.0', 'BBU_20_2.0']]  # DataFrame with two columns
    tdf = tcdf

    def buy_signals(df):
        import numpy as np
        signal = []
        for _, row in df.iterrows():
            if row['signal'] == 1:
                signal.append(row['Close'])
            else:
                signal.append(np.nan)
        return signal

    def sell_signals(df):
        import numpy as np
        signal = []
        for _, row in df.iterrows():
            if row['signal'] == -1:
                signal.append(row['Close'] * 1)
            else:
                signal.append(np.nan)
        return signal

    buy_signals = buy_signals(df)
    sell_signals = sell_signals(df)

    apds = [mpf.make_addplot(tdf),
            mpf.make_addplot(buy_signals, type='scatter', markersize=20, marker='^'),
            mpf.make_addplot(sell_signals, type='scatter', markersize=20, marker='v')]

    mpf.plot(df, type='candle', volume=True, addplot=apds)
    #mpf.plot(df.tail(100), type='candle', volume=True, mav=(10, 20), show_nontrading=False)

    # Calculate daily returns
    df['return'] = df['Close'].pct_change()

    # Calculate strategy returns
    df['strategy_return'] = df['return'] * df['signal'].shift(1)

    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1

    profit = df['cumulative_return'].iloc[-1]

    # Plot cumulative returns
    base = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

    print(f'{ticker} profit {profit} base {base}')
    return [ticker, profit, base]


def ta_bbands(ticker):
    import pandas as pd
    import ta

    # Load data
    df = yf.download(ticker, period="1mo", interval='5m', progress=False)

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

    # Identify buy and sell signals
    # df['buy_signal'] = ((df['Close'] > df['EMA5']) & (df['EMA5'] > df['EMA10']) & (df['RSI'] < 30) & (df['Close'] < df['BBL']))
    # df['sell_signal'] = ((df['Close'] < df['EMA5']) | (df['Close'] > df['BBU']))

    df['buy_signal'] = (df['RSI'] < 30)  # & (df['Close'] > df['EMA5']) & (df['EMA5'] > df['EMA10'])  # & (df['Close'] < df['BBL'])
    df['sell_signal'] = df['RSI'] > 70  # df['Close'] > df['BBU']

    # Define trading signals
    ema_buy = (df['Close'] > df['EMA5']) & (df['EMA5'] > df['EMA10'])
    ema_sell = (df['Close'] < df['EMA5']) & (df['EMA5'] < df['EMA10'])
    rsi_buy = df['RSI'] < 30
    rsi_sell = df['RSI'] > 70
    bbands_buy = df['Close'] < df['BBL']
    bbands_sell = df['Close'] > df['BBU']
    vwap_buy = (df['Close'] > df['vwap']) & (df['Close'].shift(1) <= df['vwap'].shift(1))
    vwap_sell = (df['Close'] < df['vwap']) & (df['Close'].shift(1) >= df['vwap'].shift(1))
    df['long_signal'] = bbands_buy
    df['short_signal'] = bbands_sell

    # Execute trades
    position = 0
    positions = []
    last_action = 0
    for i in range(len(df)):
        if df['long_signal'][i] and position == 0:
            position = 1
            positions.append(1)
            last_action = 1
        elif df['short_signal'][i] and position == 1:
            position = 0
            positions.append(-1)
            last_action = -1
        else:
            positions.append(0)

    # Calculate profit and loss
    df['position'] = positions

    df['pnl'] = df['position'] * df['Close'] * (-1)
    # df['pnl'] = df['position'] * (df['Close'] - df['Close'].shift(1))

    # Calculate cumulative profit and loss
    df['cumulative_pnl'] = df['pnl'].cumsum()

    # print(df)
    # df.to_csv(f'{ticker}.csv')
    profit = 0
    if last_action == 1:
        if len(df.loc[df['position'] == -1, 'cumulative_pnl']):
            profit = df.loc[df['position'] == -1, 'cumulative_pnl'].iloc[-1]
    else:
        profit = df['cumulative_pnl'].iloc[-1]

    # Plot cumulative returns
    base = df['Close'].iloc[-1] - df['Close'].iloc[0]

    def buy_signals(df):
        import numpy as np
        signal = []
        for _, row in df.iterrows():
            if row['position'] == 1:
                signal.append(row['Close'])
            else:
                signal.append(np.nan)
        return signal

    def sell_signals(df):
        import numpy as np
        signal = []
        for _, row in df.iterrows():
            if row['position'] == -1:
                signal.append(row['Close'] * 1)
            else:
                signal.append(np.nan)
        return signal

    buy_signals = buy_signals(df)
    sell_signals = sell_signals(df)

    bbs = df[['BBL', 'BBU']]  # DataFrame with two columns

    apds = [mpf.make_addplot(bbs.tail(100)),
            mpf.make_addplot(buy_signals[-100:], type='scatter', markersize=50, marker='^'),
            mpf.make_addplot(sell_signals[-100:], type='scatter', markersize=50, marker='v')]
    mpf.plot(df.tail(100), type='candle', volume=True, addplot=apds)
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
    print(df['Profit'].sum())


def vwap(ticker):
    import pandas as pd
    import pandas_ta as ta

    # Load data
    df = yf.download(ticker, period="1mo", interval='5m', progress=False)
    # df.set_index('Datetime', inplace=True)
    # Add VWAP indicator
    df['vwap'] = df.ta.vwap()
    # Define trading signals
    df['long_signal'] = (df['Close'] > df['vwap']) & (df['Close'].shift(1) <= df['vwap'].shift(1))
    df['short_signal'] = (df['Close'] < df['vwap']) & (df['Close'].shift(1) >= df['vwap'].shift(1))

    # Define trading logic
    df['position'] = 0
    df.loc[df['long_signal'], 'position'] = 1
    df.loc[df['short_signal'], 'position'] = -1
    df['position'] = df['position'].ffill()

    # Calculate returns
    df['returns'] = df['Close'].pct_change() * df['position'].shift(1)

    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['returns']).cumprod()

    # Print cumulative returns
    print(df['cumulative_returns'].iloc[-1])


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
    ta_bbands('NFLX')
