import numpy as np

from sklearn.metrics import mean_squared_error, r2_score


def multiple_variable_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.00001, no_iterations=500):

    print("\n---------------------------------------------------\n"
          "Problem 3: Linear regression with multiple features....")

    no_variables = 3
    N = len(X_train)

    w = np.zeros((no_variables + 1, 1))
    print(w.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(np.negative(X_train[:, 1]).shape)

    for i in range(no_iterations):
         for j in range(no_variables):
            w[j] = w[j] - (1 / (2 * N)) * learning_rate * (
            (y_train - (X_train.dot(w))).T.dot(np.negative(X_train[:, j])))

    y_pred = X_test.dot(w)

    print("Final values: Weight: {0}, Bias: {1}, Mean squared error: {2}, Regression score: {3} "
          .format("weight", "bias", mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

