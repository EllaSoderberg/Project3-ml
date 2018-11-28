import pandas as pd

from Problem2 import one_variable_gradient_descent

concrete_data = pd.read_csv('data/Concrete_Data.csv')
data_labels = list(concrete_data)

# Testing for all different labels (but bad results bc learning rate doesn't fit)
for label in data_labels:
    x = concrete_data['Concrete compressive strength(MPa, megapascals) ']
    y = concrete_data[label]
    one_variable_gradient_descent(x, y, 0.0001, 100)

# Testing for only one label
'''
x = concrete_data['Cement (component 1)(kg in a m^3 mixture)']
y = concrete_data['Concrete compressive strength(MPa, megapascals) ']
one_variable_gradient_descent(x, y)
'''