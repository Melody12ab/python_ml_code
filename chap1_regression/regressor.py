import sys
import numpy as np
from sklearn import linear_model

filename = sys.argv[1]
# filename = "data_singlevar.txt"
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training, 1))
# print(X_train)
Y_train = np.array(Y[:num_training])
print(Y_train)

# Test data
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:])

# Create linear regression object
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, Y_train)

import matplotlib.pyplot as plt

y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.subplot(121)
plt.scatter(X_train, Y_train, c='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')

y_test_pred = linear_regressor.predict(X_test)
plt.subplot(122)
plt.scatter(X_test, Y_test, c='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

import pickle as p

# 保存model
output_model_file = 'save_model.pkl'
with open(output_model_file, 'wb') as f:
    p.dump(linear_regressor, f)

import sklearn.metrics as sm

# 加载model
with open(output_model_file, 'rb') as f:
    model_linrgr = p.load(f)
    y_test_pred_new = model_linrgr.predict(X_test)
    print("\nNew mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred_new), 2))

# 平均绝对误差(mean absolute error) :所有数据集的所有点的绝对误差平均值
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred), 2))
# 均方误差(mean squared error):给定数据集的所有数据点的平方的平均值
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_test_pred), 2))
# 中位数绝对误差 (median absolute error):所有数据点的误差的中位数  可以消除异常值的干扰，单个坏点数据不会影响整个误差指标
print("Mean median error =", round(sm.median_absolute_error(Y_test, y_test_pred), 2))
# 解释方差分(explained variance score):衡量模型对数据集波动的解释能力 如果等分为1则表明模型很好
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_test_pred), 2))
# R2 score R方得分：衡量模型对位置样本预测效果
print("R2 score =", round(sm.r2_score(Y_test, y_test_pred), 2))
