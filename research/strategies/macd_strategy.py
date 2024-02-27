from strategy import Strategy


class MACDStrategy(Strategy):
    def signal(self):
        short_window, long_window, signal_window = 3, 7, 2
        data = self.data
        # Calculate short-term and long-term exponential moving averages
        data['Short_MA'] = data['close'].ewm(span=short_window, adjust=False).mean()
        data['Long_MA'] = data['close'].ewm(span=long_window, adjust=False).mean()

        # Calculate MACD line
        data['MACD'] = data['Short_MA'] - data['Long_MA']
        # Calculate Signal line
        data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

        # Generate Buy and Sell signals
        data['Signal'] = 0  # 0: No signal, 1: Buy, -1: Sell

        # Buy signal: MACD crosses above Signal line
        data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1

        # Sell signal: MACD crosses below Signal line
        data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1

        data.dropna(subset=['close'], inplace=True)

        return data
