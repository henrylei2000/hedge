import sqlite3
import pandas as pd


def trade(ticker, action, quantity, price):
    con = sqlite3.connect('asset_data/asset.db')
    cur = con.cursor()
    statement = f"INSERT INTO trades (ticker, action, quantity, price, at) VALUES ('{ticker}', {action}, {quantity}, {price}, datetime('now'))"
    cur.execute(statement)
    con.commit()

    amount = action * price * quantity * -1
    note = f'{ticker} position: {action}'
    statement = f"INSERT INTO transactions (amount, note, at) VALUES ({amount}, '{note}', datetime('now'))"
    cur.execute(statement)
    con.commit()
    con.close()


def get_portfolio():
    con = sqlite3.connect('asset_data/asset.db')
    statement = f"SELECT ticker, price, sum(quantity * action) FROM trades GROUP BY ticker HAVING sum(action * quantity) <> 0"
    df = pd.read_sql_query(statement, con)
    con.close()
    print(df)


def get_balance():
    con = sqlite3.connect('asset_data/asset.db')
    cur = con.cursor()
    statement = f"SELECT sum(amount) FROM transactions"
    amount = cur.execute(statement).fetchone()[0]
    con.close()
    return amount


if __name__ == "__main__":
    #trade('MCK', -1, 2, 357.10)
    #trade('AES', -1, 1, 22.45)
    print(f'{get_balance():.2f}')
    get_portfolio()