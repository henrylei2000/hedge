import sqlite3


def main():
    con = sqlite3.connect('example.db')

    cur = con.cursor()

    # Create table
    cur.execute('''CREATE TABLE IF NOT EXISTS stocks
                   (date text, action text, symbol text, quantity real, price real)''')

    # Insert a row of data
    cur.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    con.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    con.close()

    con = sqlite3.connect('example.db')
    cur = con.cursor()

    for row in cur.execute('SELECT * FROM stocks ORDER BY price'):
        print(row)


if __name__ == "__main__":
    main()
