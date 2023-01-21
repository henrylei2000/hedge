import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta


class history:
    def __init__(self):
        self.today = datetime.now()
        self.a_week_ago = self.today - relativedelta(weeks=1)
        self.a_month_ago = self.today - relativedelta(months=1)
        self.three_month_ago = self.today - relativedelta(months=90)
        self.a_year_ago = self.today - relativedelta(years=1)
        self.three_year_ago = self.today - relativedelta(years=3)


def save_data(t):
    h = history()

    try:
        data = yf.download(t, start=h.a_year_ago, end=h.today, progress=False)
        data["Date"] = data.index
        data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        data.reset_index(drop=True, inplace=True)
        data.to_csv(f"data/{t}.csv")
        print(data.tail())
        return True
    except Exception as ex:
        print("Failed on:", ex)
        return False


def arima(ticker):
    df = pd.read_csv(f"data/{ticker}.csv")
    #df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)

    # Define the training and test sets
    data = df['Close']
    #train_data = df['Close'][:int(0.8 * (len(df)))]
    #test_data = df['Close'][int(0.8 * (len(df))):]

    # Fit the ARIMA model
    p, d, q = 5, 1, 2

    # Make predictions on the test set
    model = ARIMA(data, order=(p, d, q))
    fitted = model.fit()
    print(fitted.summary())
    predictions = fitted.predict()
    print(predictions)

    # The predicted values are wrong because the data is seasonal, build a SARIMA model
    import statsmodels.api as sm

    model = sm.tsa.statespace.SARIMAX(data,
                                      order=(p, d, q),
                                      seasonal_order=(p, d, q, 12))
    model = model.fit()
    print(model.summary())

    predictions = model.predict(len(data), len(data) + 30)

    data.plot(legend=True, label="Training Data", figsize=(15, 10))
    predictions.plot(legend=True, label="Predictions")

    plt.show()



# Load the NBI stock prices data for the past 5 years
ticker = "TQQQ"
print(f"Start downloading {ticker} ...")
if save_data(ticker):
    print(f"Complete downloading {ticker} ...")
    arima(ticker)
else:
    print("something went wrong...")