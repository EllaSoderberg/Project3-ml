import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Problem1 import sklearn_gradient_descent
from Problem2 import one_variable_gradient_descent
from Problem3 import multiple_variable_gradient_descent
from Problem4 import poly_transformation


def plot_all(concrete_dataset, column_names):
    for i in range(0, 7):
        plt.figure()
        plt.scatter(concrete_dataset.values[:, -1], concrete_dataset.values[:, i], color='black')
        plt.xlabel(column_names[-1])
        plt.ylabel(column_names[i])
        plt.show()


def split_data(concrete_data, selected_variable, test_size):
    concrete_X = concrete_data.values[:, np.newaxis, selected_variable]
    concrete_y = concrete_data.values[:, -1]

    test_size = int(concrete_X.size * test_size)

    # Split the data into training/testing sets
    concrete_X_train = concrete_X[:test_size]
    concrete_X_test = concrete_X[test_size:]

    # Split the targets into training/testing sets
    concrete_Y_train = concrete_y[:test_size]
    concrete_Y_test = concrete_y[test_size:]

    return concrete_X_train, concrete_X_test, concrete_Y_train, concrete_Y_test


# Testing for only one label



# Read data
concrete_data = pd.read_csv('data/Concrete_Data.csv')
data_labels = list(concrete_data)

# Plot all features, with the features on the x-axis and the target on the y axis.
#plot_all(concrete_data, data_labels)

# Data split for one feature
selected_feature = 0

concrete_X = concrete_data.values[:, np.newaxis, selected_feature]
concrete_y = concrete_data.values[:, -1]

print(len(concrete_X), len(concrete_y))

X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    concrete_X, concrete_y, test_size=0.2, random_state=42)

concrete_X_train, concrete_X_test, concrete_Y_train, concrete_Y_test = split_data(concrete_data, selected_feature, 0.8)

# Data split for multiple features
selected_features = [0, 1, 4]

concrete_X = concrete_data.values[:, selected_features]
concrete_y = concrete_data.values[:, np.newaxis, -1]
ones = np.ones((len(concrete_X), 1))
concrete_X = np.append(concrete_X, ones, axis=1)

X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    concrete_X, concrete_y, test_size=0.2, random_state=42)

# Data split for polynomials
selected_features = [0, 1, 2, 4, 5]
degree = 2

concrete_X = concrete_data.values[:, selected_features]
concrete_y = concrete_data.values[:, np.newaxis, -1]
concrete_X = poly_transformation(concrete_X, degree)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    concrete_X, concrete_y, test_size=0.2, random_state=42)

# Problem 1
sklearn_gradient_descent(X_train_single, y_train_single, X_test_single, y_test_single,
                         concrete_data.columns[-1], concrete_data.columns[selected_feature])
# W/O random data
# sklearn_gradient_descent(concrete_X_train, concrete_Y_train, concrete_X_test, concrete_Y_test)

# Problem 2
one_variable_gradient_descent(X_train_single, y_train_single, X_test_single, y_test_single,
                              concrete_data.columns[-1], concrete_data.columns[selected_feature], 0.0000001, 1000)
# W/O random data
# one_variable_gradient_descent(concrete_X_train, concrete_Y_train, concrete_X_test, concrete_Y_test, 0.0000001, 100)
print("\n---------------------------------------------------\n"
      "Problem 3: Linear regression with multiple features....")
# Problem 3
multiple_variable_gradient_descent(X_train_multiple, y_train_multiple, X_test_multiple, y_test_multiple)
print("\n---------------------------------------------------\n"
      "Problem 4: Polynomial regression by your own gradient descent....")
# Problem 4
multiple_variable_gradient_descent(X_train_poly, y_train_poly, X_test_poly, y_test_poly, 1 / 10 ** 12, 500000)




