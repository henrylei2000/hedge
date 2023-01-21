from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
boston = load_boston()
X = boston.data
y = boston.target


# Define the selector
selector = SelectKBest(f_regression, k=10)

# Fit the selector to the data
selector.fit(X, y)

# Get the selected features
X_new = selector.transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)
# Evaluate the model on the test data
score = model.score(X_test, y_test)
print("Test score: ", score)

# Define the feature selector
selector = SelectFromModel(model, threshold='median')

# Fit the selector to the data
selector.fit(X, y)

# Get the selected features
X_new = selector.transform(X)


# Create an instance of the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the data
scaler.fit(X_new)

# Transform the data
X_scaled = scaler.transform(X_new)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print('after feature extraction.....')
print(model.coef_)
print(model.intercept_)
score = model.score(X_test, y_test)
print(f"score is {score}")

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = SVC()

# Define the grid of hyperparameters to search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Define the grid search
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters: ", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

print(best_model)
# Evaluate the model on the test data
score = best_model.score(X_test, y_test)
print("Test score: ", score)


# ----------- Food Research --------------

class FoodBank:
    def __init__(self, size=10_000):
        self.size = size
        self.df = pd.DataFrame()
        np.random.seed(123)
        self.df['age'] = np.rint(np.random.rand(self.size) * 100)
        self.df['time_in_bed'] = np.clip(np.random.normal(loc=6, scale=3, size=self.size), 2, 12)
        self.df['pct_sleeping'] = np.random.rand(self.size)
        self.df['region'] = np.random.choice(np.arange(0, 10, 2), size=self.size)
        self.df['mood'] = np.random.randn(self.size)
        self.df['food'] = np.random.choice(['pizza', 'taco', 'broccoli', 'tea', 'rice'], size=self.size)

    def data(self):
        return self.df.drop('food', axis=1)

    def target(self):
        mask = (self.df['age'] > 60)
        self.df.loc[mask, 'food'] = "tea"
        self.df.loc[(self.df['pct_sleeping'] < 0.5), 'food'] = "pizza"
        self.df.loc[(self.df['time_in_bed'] > 7), 'food'] = "broccoli"
        return self.df['food']

    def clean(self):
        mask = np.where(self.df['age'] > 99)
        print(f"haha -------- {mask[0]}")
        print(f"haha -------- {self.df.shape}")
        self.df.drop(mask[0], axis=0, inplace=True)
        print(f"haha -------- {self.df.shape}")
        self.df.fillna(0, inplace=True)
        print(f"haha - {np.any(df['age'] < 0)}")
        mask = (self.df['age'] < 0) | (self.df['age'] > 100)
        self.df.drop(self.df[mask].index, axis=0, inplace=True)


fb = FoodBank(10_000)
df = fb.df
print(df.agg({'age': ['min', 'max', 'mean']}))
fb.clean()
df = fb.df
print(df.agg({'age': ['min', 'max', 'mean'], 'pct_sleeping': ['min', 'max', 'mean']}))
X = fb.data()
y = fb.target()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
print(f"coef {logreg.coef_} and intercept {logreg.intercept_}")
y_pred = logreg.predict(X_test)
print(y_pred)
score = logreg.score(X_test, y_test)
print(score)

print("\n\n--------------------------------------------\n\n")

selector = SelectKBest(k=4)
selector.fit(X, y)
X_selected = selector.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=123)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
score = clf.score(X_test, y_test)
print(score)

print("\n\n--------------Forrest------------------\n\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=123)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)
# Performance review
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", conf_mat)
if conf_mat.all():
    print(classification_report(y_true=y_test, y_pred=y_pred))


print("\n\n--------------------- CV ---------------------\n\n")
# Create an instance of the model
logreg = LogisticRegression(solver='liblinear')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=123)
# Use 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(scores)