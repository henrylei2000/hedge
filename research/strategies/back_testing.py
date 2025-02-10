# from macd_strategy import MACDStrategy
# from wave_strategy import WaveStrategy
from raft_strategy import RaftStrategy
import alpaca_trade_api as tradeapi
from alpaca_trade_api.common import URL
import configparser

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def get_dates():
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('settings', 'API_KEY')
    secret_key = config.get('settings', 'SECRET_KEY')
    api = tradeapi.REST(api_key, secret_key,  URL('https://paper-api.alpaca.markets'), api_version='v2')
    performance = 0.0
    start_date = '2025-01-27'  # 2024-02-23 2023-07-19 2024-06-24 2023-03-09
    end_date = '2025-01-27'
    calendar = api.get_calendar(start=start_date, end=end_date)
    for day in calendar:
        daily_pnl, trades = 0, 0
        for symbol in ['TQQQ']:
            strategy = RaftStrategy(symbol=symbol, open=f"{day.date.strftime('%Y-%m-%d')} {day.open}", close=f"{day.date.strftime('%Y-%m-%d')} {day.close}")
            strategy.backtest()
            if strategy.trades:
                print(f"{day.date.strftime('%Y-%m-%d')} {symbol} {strategy.pnl:.2f} ({strategy.trades})")
            performance += strategy.pnl
            daily_pnl += strategy.pnl
            trades += strategy.trades
        if trades:
            print(f"-------------------------------------------- {daily_pnl:.2f} ({trades})")
    print(f"------------ TOTAL ------------------------- {performance:.2f}")


def back_test():
    strategy = RaftStrategy()
    strategy.backtest('offline')


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


# Example usage
if __name__ == "__main__":
    # back_test()
    get_dates()
