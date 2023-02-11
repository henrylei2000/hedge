import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(size=10_000):
    df = pd.DataFrame()
    np.random.seed(123)
    df['age'] = np.clip(np.rint(np.random.normal(loc=45, scale=45, size=size)), 0, 100)
    df['time_in_bed'] = np.clip(np.random.normal(6, 1, size), 0, 10)
    df['pct_sleeping'] = np.clip(np.random.normal(loc=0.7, scale=0.1, size=size), 0, 1)
    df['favorite_food'] = np.random.choice(['pizza', 'ice cream', 'taco'], size)
    df['hate_food'] = np.random.choice(['broccoli', 'candy corn', 'eggs'], size)
    return df


def award(row):
    if row['age'] >= 90:
        return row['favorite_food']
    if row['time_in_bed'] > 7 and row['pct_sleeping'] > 0.5:
        return row['favorite_food']
    else:
        return row['hate_food']


df = get_data(100)
df['reward'] = df.apply(award, axis=1)
print("===========================")
print(df.tail())
print("===========================")

selected = df.query('(time_in_bed > 5 and pct_sleeping > 0.7) or (age > 90)')
grouped = selected.groupby('age')

print(grouped.size().nlargest(2))

new_data = grouped.mean()
print(new_data.describe())

df['reward'] = df['hate_food']
df.loc[((df['pct_sleeping'] > 0.5) & (df['time_in_bed'] > 5) | (df['age'] > 90)), 'reward'] = df['favorite_food']
print(df)

results = pd.DataFrame(
    [
        ['loop', 3249, 345],
        ['apply', 183, 6],
        ['vectorized', 1.3, 0.1]
    ],
    columns=['type', 'mean', 'std']
)

#results.set_index('type')['mean'].plot(kind='bar', title="Time")

#plt.show()


matrix = np.ones((5, 5)) * 5

df = pd.DataFrame(matrix,columns=['a','b','c','d','e'])

new = df.iloc[:5, :5].sum().sum()

print(new)

# Create the first DataFrame
df1 = pd.DataFrame({'column_name': ['A', 'B', 'C'], 'value1': [1, 2, 3]})

# Create the second DataFrame
df2 = pd.DataFrame({'column_name': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

# Merge the two DataFrames on the 'column_name' column
merged_df = df1.merge(df2, on='column_name', how='outer')
merged_df.fillna(value=merged_df.mean(), inplace=True)
trans = merged_df.T
trans.columns=['A','B','C','D']
trans = trans.iloc[1:,:]

print(trans.columns)

for column_name, column_data in trans.iteritems():
    print(column_name, column_data.shape)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Split the array into two sub-arrays
result = np.split(arr, [2,1])
print(result)

import pandas_datareader.data as web

# Define the series code for the data you want to retrieve from FRED
series = "DGS10"

# Retrieve the data from FRED
data = web.DataReader(series, 'fred')

# Convert the data into a dataframe
df = pd.DataFrame(data)

# Print the head of the dataframe
print(df.tail(10))


df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df)
partial_df =df.iloc[:2,1:]
print(partial_df)
mean = df.mean()
print(mean)
df['city'] = ['NY', 'SF', 'NY']
print(df)
grouped = df.groupby('city').mean()
print(grouped)
mask = (df['A'] > 1) & (df['city'] == 'NY')
df_filtered = df[mask]
print(df_filtered)


class Person:
    def __init__(self, name="no_name"):
        self.name = name

    def get_name(self):
        return self.name

he = Person("Henry")
name = he.get_name()
print(name)


import requests

response = requests.get('http://184.67.98.98:8080/beamme/rest/travellers/abc123')

print(response.text)