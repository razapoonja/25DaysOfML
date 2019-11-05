import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

# Storing values in their own variables
dataset = pd.read_csv('data_preprocessing.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Filling empty values by taking mean
x_missing = X[:, 1:3]
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x_missing)
x_missing = imputer.transform(x_missing)

# Encoding values (string into numbers)
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
