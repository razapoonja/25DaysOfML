import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data pre-preprocessing
dataset = pd.read_csv('salary_data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Training
regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.predict(x_test)

# Visualizing
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
