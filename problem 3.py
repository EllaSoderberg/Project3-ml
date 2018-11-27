import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

concrete_dataset = pd.read_csv('data/Concrete_Data.csv')
collumn_names = list(concrete_dataset)


selected_variable = [0,3,4]
concrete_X = concrete_dataset.values[:, selected_variable]
concrete_Y = concrete_dataset.values[:,np.newaxis, -1]
ones = np.ones((len(concrete_X),1))
concrete_X = np.append(concrete_X, ones,axis=1)

#print(len(concrete_X))

test_size = int(len(concrete_X) * 0.8)

# Split the data into training/testing sets
concrete_X_train = concrete_X[:test_size]
concrete_X_test = concrete_X[test_size:]

#print(concrete_X_test.shape)

# Split the targets into training/testing sets
concrete_Y_train = concrete_Y[:test_size]
concrete_Y_test = concrete_Y[test_size:]

#print(concrete_Y_test.shape)

no_steps = 500
no_variables = 3
learning_rate = 0.00001
N = len(concrete_X_train)
print(N)

w = np.zeros((no_variables+1,1))
print(w.shape)
print(concrete_X_train.shape)
print(concrete_Y_train.shape)
print(np.negative(concrete_X_train[:,1]).shape)

for i in range(no_steps):
    for j in range(no_variables):
        w[j]= w[j] - (1/(2*N))*learning_rate*((concrete_Y_train-(concrete_X_train.dot(w))).T.dot(np.negative(concrete_X_train[:,j])))
        #print("w[",j,"] = ", w[j])



concrete_y_pred = concrete_X_test.dot(w)


# The bias
#print('Bias: ', regr.intercept_)
# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %f" % mean_squared_error(concrete_Y_test, concrete_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %f' % r2_score(concrete_Y_test, concrete_y_pred)) #regr.score(concrete_X_test, concrete_Y_test)

# Plot outputs
#plt.figure()

#plt.scatter(concrete_X_test, concrete_Y_test,  color='black')
#plt.plot(concrete_X_test, concrete_y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()