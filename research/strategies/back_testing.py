from macd_strategy import MACDStrategy

if __name__ == "__main__":
    # Download stock data
    stock = "TQQQ"
    # Backtest MACD strategy
    macd_strategy = MACDStrategy(stock)
    macd_strategy.backtest()
    print(f"------- Total PnL Performance ------------ {macd_strategy.pnl:.2f}")
