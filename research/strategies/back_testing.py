from macd_strategy import MACDStrategy
import alpaca_trade_api as tradeapi
import configparser


def get_dates():
    # Load Alpaca API credentials from configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Access configuration values
    api_key = config.get('settings', 'API_KEY')
    secret_key = config.get('settings', 'SECRET_KEY')
    # Initialize Alpaca API
    api = tradeapi.REST(api_key, secret_key, 'https://paper-api.alpaca.markets', api_version='v2')

    # Define the start and end dates for the market calendar you want to retrieve
    start_date = '2024-06-12'
    end_date = '2024-07-12'

    # Get the market calendar
    calendar = api.get_calendar(start=start_date, end=end_date)

    # Print the market calendar details
    performance = 0.0
    for day in calendar:
        daily_pnl, trades = 0, 0
        for symbol in ['TQQQ', 'SQQQ']:
            macd_strategy = MACDStrategy(symbol=symbol, open=f"{day.date.strftime('%Y-%m-%d')} {day.open}", close=f"{day.date.strftime('%Y-%m-%d')} {day.close}")
            macd_strategy.backtest()
            if macd_strategy.trades:
                print(f"{day.date.strftime('%Y-%m-%d')} {symbol} {macd_strategy.pnl:.2f} ({macd_strategy.trades})")
            performance += macd_strategy.pnl
            daily_pnl += macd_strategy.pnl
            trades += macd_strategy.trades
        if trades:
            print(f"-------------------------------------- {daily_pnl:.2f} ({trades})")
    print(f"------------ TOTAL ------------------- {performance:.2f}")

def back_test():

    # Backtest MACD strategy
    macd_strategy = MACDStrategy()
    macd_strategy.backtest()


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject, body, to_addr, from_addr, password):
    """
    Send an email.

    Parameters:
    - subject: Email subject
    - body: Email body content
    - to_addr: Recipient's email address
    - from_addr: Sender's email address
    - password: Sender's email password or app-specific password
    """
    # Create message object instance
    msg = MIMEMultipart()

    # Setup the parameters of the message
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    # Attach the body to the message instance
    msg.attach(MIMEText(body, 'plain'))

    # Create server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    try:
        # Login Credentials for sending the email
        server.login(from_addr, password)

        # Send the message via the server
        server.sendmail(from_addr, to_addr, msg.as_string())
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()


def email_test():
    # Replace these with your details
    SUBJECT = "Test Email"
    BODY = "This is a test email sent from a Python script."
    TO_ADDRESS = "leijin@yahoo.com"
    FROM_ADDRESS = "jinleiatyahoo@gmail.com"
    PASSWORD = "iniv srus ycjz ewpe"  # Consider using an environment variable for security

    send_email(SUBJECT, BODY, TO_ADDRESS, FROM_ADDRESS, PASSWORD)


# Example usage
if __name__ == "__main__":
    # back_test()
    get_dates()
