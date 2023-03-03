import pandas as pd
import mplfinance as mpf
import numpy as np
import ta  # https://github.com/bukosabino/ta/
from bs4 import BeautifulSoup
import requests


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


def fund_estimate(tickers):
    estimates = pd.DataFrame([], columns=['Ticker', 'Consensus', 'Category', 'To Target'])
    for ticker in tickers:
        try:
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
                # print(f'------------{t}------------')
                # print(df.head())

            tables = ['quarterly number', 'Estimates']

            for t in tables:
                # print(f'------------{t}------------')
                df = get_table(soup, t)
                # print(df.head())

            # print(f'------------Ratings------------')
            df = get_ratings(soup)
            consensus = ':'.join(df['Current'].astype(str))
            # print(df.head())

            # print(f'------------Updates------------')
            df = get_updates(soup)
            # print(df.head())

            row = pd.DataFrame({
                'Ticker': [ticker],
                'Consensus': [consensus],
                'Category': [recommendation],
                'Target': [median]
            })

            estimates = pd.concat([estimates, row])
        except Exception as e:
            pass
    return estimates