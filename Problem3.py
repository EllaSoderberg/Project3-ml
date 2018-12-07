import numpy as np

from sklearn.metrics import mean_squared_error, r2_score


def multiple_variable_gradient_descent(X_train, y_train, X_test, y_test, learning_rate=0.00001, no_iterations=500):

    no_variables = X_train[0].size - 1
    N = len(X_train)

    w = np.zeros((no_variables + 1, 1))

    for i in range(no_iterations):
         for j in range(no_variables):
            w[j] = w[j] - (1 / (2 * N)) * learning_rate * (
            (y_train - (X_train.dot(w))).T.dot(np.negative(X_train[:, j])))

    #y_pred_train = X_train.dot(w)
    y_pred = X_test.dot(w)

    #print("Training values: Mean squared error: {0}, Regression score: {1} "
    #      .format(mean_squared_error(y_train, y_pred_train), r2_score(y_test, y_pred_train)))

    print("Final values: Mean squared error: {0}, Regression score: {1} "
          .format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

