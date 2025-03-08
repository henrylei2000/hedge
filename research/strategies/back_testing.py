from candle_strategy import CandleStrategy
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from alpaca_trade_api.common import URL
import configparser
import pandas as pd

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def test(mode='online'):
    if mode == 'online':
        start_date = '2024-03-06'  # 2024-02-23 2023-07-19 2024-06-24 2023-03-09
        end_date = '2024-03-06'
        symbols = ['TQQQ']
        config = configparser.ConfigParser()
        config.read('config.ini')
        api_key = config.get('settings', 'API_KEY')
        secret_key = config.get('settings', 'SECRET_KEY')
        api = tradeapi.REST(api_key, secret_key,  URL('https://paper-api.alpaca.markets'), api_version='v2')
        performance = 0.0
        calendar = api.get_calendar(start=start_date, end=end_date)

        context = {}
        end_date = pd.to_datetime(start_date) - pd.DateOffset(days=1)
        start_date = end_date - pd.DateOffset(months=3)
        for symbol in symbols:
            start = start_date.strftime('%Y-%m-%d')
            end = end_date.strftime('%Y-%m-%d')
            bars = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start, end).df
            context[symbol] = bars

        for day in calendar:
            daily_pnl, trades = 0, 0
            for symbol in symbols:
                current = day.date.strftime('%Y-%m-%d')
                strategy = CandleStrategy(symbol, f"{current} {day.open}", f"{current} {day.close}", api, context[symbol])
                strategy.backtest()
                if strategy.trades:
                    print(f"{day.date.strftime('%Y-%m-%d')} {symbol} {strategy.pnl:.2f} ({strategy.trades})")
                performance += strategy.pnl
                daily_pnl += strategy.pnl
                trades += strategy.trades
            if trades:
                print(f"-------------------------------------------- {daily_pnl:.2f} ({trades})")
        print(f"------------ TOTAL ------------------------- {performance:.2f}")
    else:
        strategy = CandleStrategy()
        strategy.backtest()
        print(f"-------------------------------------------- {strategy.pnl:.2f} ({strategy.trades})")


def notification():
    subject = "Test Email"
    body = "This is a test email sent from a Python script."
    config = configparser.ConfigParser()
    config.read('config.ini')
    to_address = config.get('email', 'TO_ADDRESS')
    from_address = config.get('email', 'FROM_ADDRESS')
    url = config.get('email', 'URL')
    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    try:
        server.login(from_address, url)
        server.sendmail(from_address, to_address, msg.as_string())
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()


if __name__ == "__main__":
    test('online')
