import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

concrete_dataset = pd.read_csv('data/Concrete_Data.csv')
deg = 3

def polyFeatures(data, degree, p):
    values = []

    if degree > 1:
        values = polyFeatures(data, degree - 1, p)
        k = len(values)

        if degree > 2:
            for i in range(1, len(data)):
                p[len(data) - 1 - i] = p[len(data) - 1 - i] + p[len(data) - i]

        for i in range(0, len(data)):
            add = k - p[i]
            for j in range(add, k):
                values.append(data[i] * values[j])

    else:
        values.append(1)

        for i in range(0, len(data)):
            values.append(data[i])
    return values

def polyTransormation(data, degree):
    newdata = []

    for j in range(0, len(data)):
        p = []
        for i in range(len(data[j]), 0, -1):
            p.append(i)

        values = polyFeatures(data[j], degree, p)
        newdata.append(values)

    return np.array(newdata)

concrete_X = concrete_dataset.values[:, [3, 2, 1]]
#Build in
#poly = PolynomialFeatures(degree=deg)
#concrete_X = poly.fit_transform(concrete_X)
concrete_X = polyTransormation(concrete_X, deg)
concrete_Y = concrete_dataset.values[:, -1]

test_size = int(concrete_Y.size * 0.8)

# Split the data into training/testing sets
concrete_X_train = concrete_X[:test_size]
concrete_X_test = concrete_X[test_size:]

# Split the targets into training/testing sets
concrete_Y_train = concrete_Y[:test_size]
concrete_Y_test = concrete_Y[test_size:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(concrete_X_train, concrete_Y_train)

# Make predictions using the testing set
concrete_y_pred = regr.predict(concrete_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %f"
% mean_squared_error(concrete_Y_test, concrete_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %f' % r2_score(concrete_Y_test, concrete_y_pred))
