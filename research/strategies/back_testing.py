from macd_strategy import MACDStrategy


def back_test():

    # Backtest MACD strategy
    macd_strategy = MACDStrategy()
    macd_strategy.backtest()
    print(f"------- Total PnL Performance ------------ {macd_strategy.pnl:.2f}")



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
    back_test()
