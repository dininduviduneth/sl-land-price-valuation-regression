# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Dinindu Seneviratne/OneDrive - nCinga Innovations Pte Ltd/Part Time Study/Research Study/Implementation/Model Test - 31082020 - Colombo District Only - Without Scenic View and Highway.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Not Required to Dummify in this scenario

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

import math
rmse = math.sqrt(mse)
print(rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2)