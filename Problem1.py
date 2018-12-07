import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def sklearn_gradient_descent(X, y, X_test, y_test, x_label=None, y_label=None, plot=True):

    print("\n---------------------------------------------------\nProblem 1: Linear regression using sklearn....")

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, y)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # Print the weight, the bias, the mean squared error and the variance score (where 1 is perfect prediction)
    print("Final values: Weight: {0}, Bias: {1}, Mean squared error: {2}, Regression score: {3} "
          .format(regr.coef_, regr.intercept_, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

    # Plot outputs
    if plot:
        plt.figure()

        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
