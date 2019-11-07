import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


# Building the optimal model using Backward Elimination

# Adding column of 1's
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

x_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = x_opt).fit() # OLS = ordinary least squares
print(regressor_OLS.summary())
x_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = x_opt).fit()
print(regressor_OLS.summary())
x_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = x_opt).fit()
print(regressor_OLS.summary())
x_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = x_opt).fit()
print(regressor_OLS.summary())
x_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = x_opt).fit()
print(regressor_OLS.summary())