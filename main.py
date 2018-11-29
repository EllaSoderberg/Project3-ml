import pandas as pd
import numpy as np


from Problem2 import one_variable_gradient_descent

concrete_data = pd.read_csv('data/Concrete_Data.csv')
data_labels = list(concrete_data)

selected_variable = 0
concrete_X = concrete_data.values[:, np.newaxis, selected_variable]
concrete_Y = concrete_data.values[:, -1]

test_size = int(concrete_X.size * 0.8)

# Split the data into training/testing sets
concrete_X_train = concrete_X[:test_size]
concrete_X_test = concrete_X[test_size:]

# Split the targets into training/testing sets
concrete_Y_train = concrete_Y[:test_size]
concrete_Y_test = concrete_Y[test_size:]

'''
# Testing for all different labels (but bad results bc learning rate doesn't fit)
for label in data_labels:
    x = concrete_data['Concrete compressive strength(MPa, megapascals) ']
    y = concrete_data[label]
    one_variable_gradient_descent(x, y, 0.0001, 100)

# Testing for only one label
'''
x = concrete_data['Cement (component 1)(kg in a m^3 mixture)']
y = concrete_data['Concrete compressive strength(MPa, megapascals) ']
one_variable_gradient_descent(concrete_X_train, concrete_Y_train, concrete_X_test, concrete_Y_test, 0.00001, 100)
