"""
Analyze the analysis
"""
from bs4 import BeautifulSoup
import requests
import pandas as pd


def get_two_columns(soup, type):
    table = soup.find('table', {'aria-label': f'{type} data table'})
    rows = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
        rows.append(row_data)
    df = pd.DataFrame(rows)
    return df


def get_table(soup, type):
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

def get_table2(soup, type):
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

url = "https://www.marketwatch.com/investing/stock/tsla/analystestimates"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tables = ['snapshot', 'stock price targets']

for t in tables:
    df = get_two_columns(soup, t)
    print(f'------------{t}------------')
    print(df.head())


tables = ['quarterly number']

for t in tables:
    df = get_table(soup, t)
    print(df.head())

tables = ['upgrades/downgrades']

for t in tables:
    df = get_table2(soup, t)
    print(df.head())

