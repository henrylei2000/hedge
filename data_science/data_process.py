from sklearn import datasets
import pandas as pd
import numpy as np
import re

a = np.array([1, 2, 3, 4, 5])
data = pd.Series(a)
target = pd.Series(['a', 'b', 'c', 'd', 'e'])

df = pd.concat([data, target], axis=1)
df.columns = ['data', 'target']
print(df)

print("The df structure: %d x %d %d" % (df.shape[0], df.shape[1], df.size))
iris = datasets.load_iris()

iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_target = pd.DataFrame(iris.target, columns=['target'])

iris_df = pd.concat([iris_data, iris_target], axis=1)


# Create a DataFrame from a dictionary
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie', "David"],
                   'Age': [45, 32, 19, 35],
                   'City': ['New York', 'San Francisco', 'Los Angeles', 'New York']})

# Add a new column to the DataFrame
df['Salary'] = [50000, 60000, 40000, 80000]

# Filter rows where age is greater than 30
filtered_df = df.loc[(df['Age'] > 30) & (df['Salary'] > 50000)]

# Print the filtered DataFrame

# Select only the 'Name' and 'Age' columns
selected_df = df[['Name', 'Age']]

# Print the selected DataFrame
print(selected_df)

# Group the DataFrame by city and calculate the mean salary
grouped_df = df.groupby('City').mean()
# Print the grouped DataFrame
# print(grouped_df)

# Create a DataFrame
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
                   'B': ['one', 'one', 'two', 'two', 'two', 'one'],
                   'C': [1, 2, 3, 4, 5, 6],
                   'D': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

# Group the DataFrame by column 'A'
grouped = df.groupby('A').agg({'C': ['mean', 'std'], 'D': ['mean', 'std']})

# Print the result
print(grouped)

# Create the first DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                   'value': [1, 2, 3, 4]})

# Create the second DataFrame
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
                   'value': [5, 6, 7, 8]})

# Merge the DataFrames on the 'key' column
merged_df = pd.merge(df1, df2, on='key', how="left")
merged_df['value_y'].fillna(merged_df['value_y'].mean(), inplace=True)

# Create a DataFrame
df = pd.DataFrame({'A': [1, 1, 2, 4],
                   'B': [5, 5, 7, 8],
                   'C': [9, 9, 11, 12]})

# Group by column 'A' and calculate the mean of column 'C' for each group
df = df.drop_duplicates(keep=False)
#df['duplicated'] = df.duplicated()

# Print the result
print(df)

print("--------------------------")
# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
                   'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 30]})

# Calculate the mean and standard deviation of column 'A'
mean = df['A'].mean()
std = df['A'].std()
print(std)
# Remove any rows with a value greater than 3 standard deviations from the mean
df[(np.abs(df['A'] - mean) > 2 * std)] = mean + 2 * std


# Create a DataFrame
df = pd.DataFrame({'A': ['John Smith', 'Jane Smith', 'Bob Jones'],
                   'B': ['111-222-3333', '222-333-4444', '333-444-5555']})

# Extract the first name from column 'A' using regex


def first_name(dfA):
    return dfA.split()[0]


df['A'] = df['A'].apply(lambda dfa: dfa.split()[0])


# Print the result
print(df)

print("--------------------------")
print("--------------------------")

# Create two 1-dimensional arrays
a = np.array([1, 2, 3])
b = np.array((4, 5, 6))

# Concatenate the arrays along the rows
c = np.concatenate((a, b))
c = np.ones(6).reshape(2,3)
c = np.arange(2, 11, 2)
c = np.linspace(0, 1, 10)
# Print the concatenated array

df = pd.DataFrame([a, b], columns=['A', 'B', 'C'])
print(df)
# Split the array into two parts
d, e = np.split(c, [3])

#print(np.ones(10)*5)


# Create an array with some missing values
a = np.array([1, 2, np.nan, 4, 5])

# Replace missing values with 0
#print(np.isnan(a))
a[np.isnan(a)] = -1


df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                   'B': [5, np.nan, 7, np.nan],
                   'C': [9, 10, 11, 12]})

# Drop rows with missing values
# df = df.dropna()
# df = df.fillna(0)
df = df.interpolate()
# Print the result


# time series

# Create a DataFrame
df = pd.read_csv('../quant/data/A.csv', parse_dates=['Date'], index_col='Date')
#dw = df.resample('W').mean()
df['rolling_mean'] = df['Adj Close'].rolling(window=10).mean()

df['rolling_mean'].fillna(value=np.nan, inplace=True)

# Print the result
print(df[['Low', 'High', 'Adj Close', 'rolling_mean']])

text = ['aQx 12', 'aub 6 5']
df = pd.DataFrame({'name': text})
df['name'] = df['name'].apply(lambda x: x.split(' ')[0])

