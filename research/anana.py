"""
Analyze the analysis
"""
from bs4 import BeautifulSoup
import requests
import pandas as pd

import os

def html_table_dataframe(url, idx):
    r = requests.get(url)
    # Parsing the HTML
    soup = BeautifulSoup(r.content, 'html.parser')
    tables = [
        [
            [td.get_text(strip=True) for td in tr.find_all('td')]
            for tr in tab.find_all('tr')
        ]
        for tab in soup.find_all('table')
    ]

    return pd.DataFrame(tables[idx])
def tmt_from_nasdaq100():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    idx = 4
    df = html_table_dataframe(url, idx)
    df.dropna(inplace=True)
    df.columns = ['Company', 'Ticker', 'GICS Sector', 'GICS Sub-Industry']
    print(df['Company'])

    # Define the condition
    condition = (df['GICS Sector'] != 'Health Care') & \
                (df['GICS Sector'] != 'Utilities') & \
                (df['GICS Sector'] != 'Industrials') & \
                (df['GICS Sector'] != 'Energy') &\
                (df['GICS Sector'] != 'Consumer Staples') & \
                (df['GICS Sector'] != 'Consumer Discretionary') & \
                (df['Ticker'] != 'GOOGL')

    exception = (df['GICS Sub-Industry'] == 'Internet & Direct Marketing Retail') |\
                (df['Ticker'] == 'TSLA')

    # Use the condition to select rows from the DataFrame
    tmt_rows = df[condition | exception]
    tmt_rows.to_csv("tmt_nasdaq100.csv")

def get_tmt_tickers():
    file = "tmt_nasdaq100.csv"
    if not os.path.exists(file):
        tmt_from_nasdaq100()
    df = pd.read_csv(file)
    return df['Ticker']


def get_two_columns(soup, type):
    table = soup.find('table', {'aria-label': f'{type} data table'})
    rows = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
        rows.append(row_data)
    df = pd.DataFrame(rows, columns=['key', 'value'])
    df.set_index(['key'], inplace=True)
    return df


def get_table(soup, type):
    if type == 'Estimates':
        div = soup.find('div', {'data-tab-pane': 'Estimates'})
        table = div.find('table')
    else:
        table = soup.find('table', {'aria-label': f'{type} data table'})
    header_row = table.find('thead').find('tr')
    headers = [header.text.strip() for header in header_row.find_all('th')]
    body = table.find('tbody')
    rows = []
    for row in body.find_all('tr'):
        row_data = []
        index = row.find('td')
        row_data.append(index.text.strip())
        for cell in row.find_all('th'):
            row_data.append(cell.text.strip())
        rows.append(row_data)

    df = pd.DataFrame(rows, columns=headers)
    return df

def get_updates(soup):
    type = 'upgrades/downgrades'
    table = soup.find('table', {'aria-label': f'{type} data table'})
    header_row = table.find('thead').find('tr')
    headers = [header.text.strip() for header in header_row.find_all('th')]
    body = table.find('tbody')
    rows = []
    for row in body.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
        rows.append(row_data)

    df = pd.DataFrame(rows, columns=headers)
    return df

def get_ratings(soup):
    element = soup(text="Analyst Ratings")
    table = element[0].parent.parent.parent.next_sibling.next_sibling
    header_row = table.find('thead').find('tr')
    headers = [header.text.strip() for header in header_row.find_all('th')]
    body = table.find('tbody')
    rows = []
    for row in body.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
        rows.append(row_data)

    df = pd.DataFrame(rows, columns=headers)
    return df


recommendations = pd.DataFrame([], columns=['Ticker', 'Recommendation', 'Gap to Estimated'])

# tickers = get_tmt_tickers()
# df = pd.read_csv('xtmt_nasdaq100.csv')
# tickers = df['Ticker']
# tickers = ['AI', 'SOUN', 'GOOG', 'MSFT', 'AMZN', 'AAPL']
tickers = get_tmt_tickers()

print(len(tickers))

for ticker in tickers:
    try:
        print(ticker)
        url = f'https://www.marketwatch.com/investing/stock/{ticker}/analystestimates'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')


        tables = ['snapshot', 'stock price targets']

        for t in tables:
            df = get_two_columns(soup, t)
            if t == 'snapshot':
                recommendation = df.loc['Average Recommendation', 'value']
            if t == 'stock price targets':
                current_price_string = df.loc['Current Price', 'value'][1:].replace(',', '')
                current_price = float(current_price_string)
                median_string = df.loc['Median', 'value'][1:].replace(',', '')
                median = float(median_string)
                diff = (current_price - median) / median * 100
            print(f'------------{t}------------')
            print(df.head())

        tables = ['quarterly number', 'Estimates']

        for t in tables:
            print(f'------------{t}------------')
            df = get_table(soup, t)
            print(df.head())

        print(f'------------Ratings------------')
        df = get_ratings(soup)
        print(df.head())

        print(f'------------Updates------------')
        df = get_updates(soup)
        print(df.head())

        print('\n\n\n')

        row = pd.DataFrame({
            'Ticker': [ticker],
            'Recommendation': [recommendation],
            'Gap to Estimated': [diff]
        })

        recommendations = pd.concat([recommendations, row])
    except Exception as e:
        pass

res = recommendations.sort_values('Gap to Estimated')
res.set_index(['Ticker'], inplace=True)
print(res)

