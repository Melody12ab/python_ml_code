import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

filename = sys.argv[1]
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        Y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# X_train = np.array(X[:num_training]).reshape((num_training, 1))
X_train = np.array(X[:num_training])
Y_train = np.array(Y[:num_training])
X_test = np.array(X[num_training:])
Y_test = np.array(Y[num_training:])

linear_regressor = linear_model.LinearRegression()
# ridge regression alpha用来控制回归器的复杂度，趋于0时，使用普通最小二乘，如需对异常值不敏感，应当设置大点
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

linear_regressor.fit(X_train, Y_train)
ridge_regressor.fit(X_train, Y_train)

y_test_pred = linear_regressor.predict(X_test)
y_test_pred_ridge = ridge_regressor.predict(X_test)

print("Linear:\n")
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_test_pred), 2))
print("Mean median error =", round(sm.median_absolute_error(Y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, y_test_pred), 2))

print("Ridge:\n")
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred_ridge), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_test_pred_ridge), 2))
print("Mean median error =", round(sm.median_absolute_error(Y_test, y_test_pred_ridge), 2))
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_test_pred_ridge), 2))
print("R2 score =", round(sm.r2_score(Y_test, y_test_pred_ridge), 2))

from sklearn.preprocessing import PolynomialFeatures

# degree 控制表示使用多少项的多项式
polynomial = PolynomialFeatures(degree=5)
X_train_transformed = polynomial.fit_transform(X_train)
# 必须是一个二维数组
datapoint = [[0.39, 2.78, 7.11]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, Y_train)

print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

# Stochastic Gradient Descent regressor
sgd_regressor = linear_model.SGDRegressor(loss='huber', max_iter=50)
sgd_regressor.fit(X_train, Y_train)
print("\nSGD regressor: \n", sgd_regressor.predict(datapoint))
