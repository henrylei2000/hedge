import sqlite3


def trade(ticker, action, quantity, price):
    con = sqlite3.connect('asset_data/asset.db')
    cur = con.cursor()
    statement = f"INSERT INTO trades (ticker, action, quantity, price, at) VALUES ('{ticker}', {action}, {quantity}, {price}, datetime('now'))"
    cur.execute(statement)
    con.commit()

    amount = action * price * quantity * -1
    note = f'{ticker} position: {action}'
    statement = f"INSERT INTO transactions (amount, at) VALUES ('{amount}', {note}, datetime('now'))"
    cur.execute(statement)
    con.commit()


def get_portfolio():
    con = sqlite3.connect('asset_data/asset.db')
    cur = con.cursor()
    statement = f"SELECT * FROM trades"
    trades = cur.execute(statement)
    for t in trades:
        print(t)
    con.commit()


def get_balance():
    con = sqlite3.connect('asset_data/asset.db')
    cur = con.cursor()
    statement = f"SELECT sum(amount) FROM transactions"
    amount = cur.execute(statement).fetchone()[0]
    return amount


if __name__ == "__main__":
    get_balance()
