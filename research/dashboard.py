import numpy as np
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import altair as alt
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


def tmt_from_nasdaq100(file):
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
    tmt_rows.to_csv(file)


def get_tmt_tickers(file):
    if not os.path.exists(file):
        tmt_from_nasdaq100(file)
    df = pd.read_csv(file)
    return df['Ticker']


def read_info(ticker):
    # Making a GET request
    r = requests.get(f'https://www.marketwatch.com/investing/stock/{ticker}/company-profile')

    # Parsing the HTML
    soup = BeautifulSoup(r.content, 'html.parser')
    s = soup.find('p', class_='description__text')
    if s:
        return s.text
    else:
        return ""


def get_pe(ticker):
    pe = -1
    # Making a GET request
    url = f'https://www.google.com/finance/quote/{ticker}:NASDAQ'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    element = soup(text="P/E ratio")
    if len(element):
        tag = element[0].parent.parent.next_sibling
        if tag.text != '-':
            t = tag.text.replace(',', '')
            pe = float(t)
    cap = 0
    element = soup(text="Market cap")
    if len(element):
        tag = element[0].parent.parent.next_sibling
        if tag.text.endswith('B USD'):
            cap = float(tag.text[:-5])
        elif tag.text.endswith('T USD'):
            cap = float(tag.text[:-5]) * 1000

    return pe, cap


def save_data(tickers):
    for t in tickers:
        print(t)
        try:
            # get the historical prices for this ticker
            df = yf.Ticker(t).history(period='90d')
            df.to_csv(f"stock_data/{t}.csv")
        except Exception as ex:
            print("Failed on:", ex)


def normalize(data):
    pass


def compare_stocks(tickers):
    data = pd.DataFrame([], columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock '
                                                                                                              'Splits'])
    for t in tickers:
        df = pd.read_csv(f'./stock_data/{t}.csv')
        df['Ticker'] = t
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['Close'] = df['Close'] / df['Close'][0] * 100

        data = pd.concat([data, df], axis=0)

    lines = (
        alt.Chart(data, height=600, title="ChatGPT Disruption")
        .mark_line()
        .encode(
            x=alt.X("Date", title="Date"),
            y=alt.Y("Close", title="Price"),
            color="Ticker",
        )
    )
    st.altair_chart(lines.interactive(), use_container_width=True)

def present(tickers):
    st.set_page_config(
        page_title="TMT Research",
        page_icon="https://www.svbsecurities.com/wp-content/uploads/2022/04/svb_fav-2-32x32.png"
    )
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.write('# TMT Nasdaq100')

    df_fin = pd.DataFrame([], columns=['Ticker', 'PERatio', 'MarketCapital'])
    df_cap = pd.DataFrame([], columns=['Ticker', 'MarketCapital'])

    for t in tickers:
        pe, cap = get_pe(t)
        df_fin.loc[len(df_fin)] = [t, pe, cap]

    stable_df = df_fin[df_fin['PERatio'] > 0]
    pe_mean = round(stable_df['PERatio'].mean(), 2)
    st.metric('P/E Ratio Avg (earnings > 0)', pe_mean)

    df_fin.sort_values(by='PERatio', inplace=True)

    st.write(alt.Chart(df_fin).mark_bar().encode(
        x=alt.X('Ticker', sort=None),
        y='PERatio',
        color=alt.condition(
            alt.datum.PERatio > 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).properties(width=780))

    cap_mean = round(df_fin['MarketCapital'].mean(), 2)
    st.metric('Market Capital Avg', f'{cap_mean}B')

    df_fin.sort_values(by='MarketCapital', inplace=True)

    st.write(alt.Chart(df_fin).mark_bar().encode(
        x=alt.X('Ticker', sort=None),
        y='MarketCapital',
        color= alt.value("olive")
    ).properties(width=780))


    chart_data = pd.DataFrame(
        df_fin,
        columns=['Ticker', 'PERatio', 'MarketCapital'])

    c = alt.Chart(chart_data).mark_circle().encode(
        x='Ticker', y='PERatio', size='MarketCapital', color='MarketCapital', tooltip=['Ticker', 'PERatio', 'MarketCapital'])

    st.altair_chart(c, use_container_width=True)

file = 'tmt_nasdaq100.csv'
clusters = get_tmt_tickers(file)
present(clusters)

ai_clusters = ['AI', 'SOUN', 'GOOG', 'MSFT', 'AMZN', 'AAPL']
save_data(ai_clusters)
compare_stocks(ai_clusters)