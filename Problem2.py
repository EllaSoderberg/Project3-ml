
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

"""
Functions to find starting coefficients that brings fairly
accurate results based on the variance and covariance of the data.
"""


def variance(values):
    """
    Calculates the variance of a set of integers
    :param values: an array of integers
    :return: the variance
    """
    total = 0
    mean_square = values.mean() ** 2
    for val in values:
        var = val - mean_square
        total += var
    return total


def covariance(x_vals, y_vals):
    """
    Calculates the covariance of two sets of integers
    :param x_vals, y_vals: arrays of integers
    :return: the covariance
    """
    covar = 0
    mean_x = x_vals.mean()
    mean_y = y_vals.mean()
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    for i in range(len(x_vals)):
        total = (x_vals[i] - mean_x) * (y_vals[i] - mean_y)
        covar += total
    return covar


def coefficient_estimation(x_vals, y_vals):
    """
    A function to estimate the weight and bias for the function y = weight * x + bias
    :param x_vals, y_vals: arrays of integers
    :return: estimated weight and bias
    """
    weight = covariance(x_vals, y_vals)/variance(x_vals)
    bias = y_vals.mean() - (weight * x_vals.mean())
    return weight, bias


"""
Functions to help estimate the linear regression using gradient descent
"""


def gradient_descent(weight, bias, x_vals, y_vals, learning_rate):
    """
    A function to calculate the gradient descent of two arrays of integers
    :param weight: the estimated weight as a float
    :param bias: the estimated bias as a float
    :param x_vals, y_vals: arrays of integers
    :param learning_rate: the learning rate as a float
    :return: the new weight and bias
    """
    w_grad = 0
    b_grad = 0
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    n = len(x_vals)
    for i in range(n):
        x = x_vals[i]
        y = y_vals[i]
        w_grad += -(2/n) * (y - ((bias * x) + weight))
        b_grad += -(2/n) * x * (y - ((bias * x) + weight))
    weight -= (learning_rate * w_grad)
    bias -= (learning_rate * b_grad)
    return weight, bias


def coefficient_finder(x_vals, y_vals, weight, bias, learning_rate, no_iterations):
    """
    A function to loop the function gradient_descent no_iterations times.
    :param x_vals, y_vals: arrays of integers
    :param initial_weight: the initial weight as a float
    :param initial_bias: the initial bias as a float
    :param learning_rate: the learning rate as a float
    :param no_iterations: number of iterations gradient_descent should be run
    :return: final weight and bias
    """
    for j in range(no_iterations):
        weight, bias = gradient_descent(weight, bias, x_vals, y_vals, learning_rate)
    return -weight, bias


def estimate_values(x_vals, y_vals, weight, bias):
    """
    A function to estimate the new y values depending on the weight and bias
    :param x_vals, y_vals: arrays of integers
    :param weight: the estimated weight as a float
    :param bias: the estimated bias as a float
    :return: a list of the estimated y values
    """
    estimation = []
    x_vals = list(x_vals)
    for i in range(len(y_vals)):
        estimation.append(weight * x_vals[i] + bias)
    return estimation


def one_variable_gradient_descent(x, y, x_test, y_test, x_label=None, y_label=None, learning_rate=0.0001, no_iterations=100, plot=True):

    print("\n---------------------------------------------------\nProblem 2: Linear regression using our own model....")

    initial_weight, initial_bias = coefficient_estimation(x, y)
    y_initial_pred = estimate_values(x_test, y_test, initial_weight, initial_bias)
    print("Starting values: Weight: {0}, Bias: {1}, Mean squared error: {2}, Regression score: {3}"
          .format(initial_weight, initial_bias, mean_squared_error(y_test, y_initial_pred), r2_score(y_test, y_initial_pred)))
    print("Finding a better weight and bias using gradient descent....")
    print("Learning rate is {0} and number of iterations is {1}".format(learning_rate, no_iterations))
    weight, bias = coefficient_finder(x, y, initial_weight, initial_bias, learning_rate, no_iterations)
    y_pred = estimate_values(x_test, y_test, weight, bias)
    print("Final values: Weight: {0}, Bias: {1}, Mean squared error: {2}, Regression score: {3} "
          .format(weight, bias, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax3 = fig.add_subplot(111)

        ax1.scatter(x_test, y_test, color='black')
        ax2.plot(x_test, y_initial_pred, color='black', linewidth=3)
        ax3.plot(x_test, y_pred, color='blue', linewidth=3)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()


