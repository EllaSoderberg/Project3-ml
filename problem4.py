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

def test(concrete_dataset, values, deg, steps, rate):

    selected_variable = values
    concrete_X = concrete_dataset.values[:, selected_variable]
    concrete_Y = concrete_dataset.values[:,np.newaxis, -1]
    concrete_X = polyTransormation(concrete_X, deg)
    test_size = int(len(concrete_X) * 0.8)

    # Split the data into training/testing sets
    concrete_X_train = concrete_X[:test_size]
    concrete_X_test = concrete_X[test_size:]

    # Split the targets into training/testing sets
    concrete_Y_train = concrete_Y[:test_size]
    concrete_Y_test = concrete_Y[test_size:]

    no_steps = steps
    no_variables = concrete_X[0].size - 1
    learning_rate = rate
    N = len(concrete_X_train)

    w = np.zeros((no_variables+1,1))

    for i in range(no_steps):
        for j in range(no_variables):
            w[j]= w[j] - (1/(2*N))*learning_rate*((concrete_Y_train-(concrete_X_train.dot(w))).T.dot(np.negative(concrete_X_train[:,j])))
            #print("w[",j,"] = ", w[j])



    concrete_y_pred = concrete_X_test.dot(w)
    #print(concrete_y_pred)

    # The bias
    #print('Bias: ', regr.intercept_)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    #print("Mean squared error: %f" % mean_squared_error(concrete_Y_test, concrete_y_pred))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %f' % r2_score(concrete_Y_test, concrete_y_pred)) #regr.score(concrete_X_test, concrete_Y_test)
    return mean_squared_error(concrete_Y_test, concrete_y_pred), r2_score(concrete_Y_test, concrete_y_pred)

best_mse, max_r2 = test(concrete_dataset, [0, 1, 6], 2, 500, 0.000000001)
best_values = [0, 1, 6]
best_deg = 2
best_steps = 500
best_rate = 0.00000001


for i in range(500, 1100, 100):
    for j in range(10, 30):
        mse, r2 = test(concrete_dataset, [0,1,6], 2, i, 1 / 10**j)
        if max_r2 < r2:
            max_r2 = r2
            best_mse = mse
            best_values = [0,1,6]
            best_deg = 2
            best_steps = i;
            best_rate = 1 / 10**j

print(max_r2)
print(best_mse)
print(best_values)
print(best_deg)
print(best_steps)
print(best_rate)




'''
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
'''