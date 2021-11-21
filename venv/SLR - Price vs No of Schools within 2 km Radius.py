# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Dinindu Seneviratne/OneDrive - nCinga Innovations Pte Ltd/Part Time Study/Research Study/Implementation/SLR Models/price vs no of schools in 2km radus.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price per Perch vs # of Schools within 2 km (Training set)')
plt.xlabel('# of Schools')
plt.ylabel('Price (LKR per perch)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price per Perch vs # of Schools within 2 km (Test set)')
plt.xlabel('# of Schools')
plt.ylabel('Price (LKR per perch)')
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

import math
rmse = math.sqrt(mse)
print(rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2)