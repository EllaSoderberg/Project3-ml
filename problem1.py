import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

concrete_dataset = pd.read_csv('data/Concrete_Data.csv')
collumn_names = list(concrete_dataset)

print(concrete_dataset.head(5))

for i in range(0, 7):
    plt.figure()
    plt.scatter(concrete_dataset.values[:, i], concrete_dataset.values[:, -1], color='black')
    plt.xlabel(collumn_names[i])
    plt.ylabel(collumn_names[-1])
    plt.show

selected_variable = 0;
concrete_X = concrete_dataset.values[:, np.newaxis, selected_variable]
concrete_Y = concrete_dataset.values[:, -1]

test_size = int(concrete_X.size * 0.8)

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

# The bias
print('Bias: ', regr.intercept_)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %f" % mean_squared_error(concrete_Y_test, concrete_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %f' % r2_score(concrete_Y_test, concrete_y_pred)) #regr.score(concrete_X_test, concrete_Y_test)

# Plot outputs
plt.figure()

plt.scatter(concrete_X_test, concrete_Y_test,  color='black')
plt.plot(concrete_X_test, concrete_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()