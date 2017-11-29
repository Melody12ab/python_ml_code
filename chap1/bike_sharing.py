import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_feature_importances(feature_importances, title, feature_names):
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # flipud 翻转，从大到小排   argsort 返回排序后的index
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0]) + 0.5

    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:14])
        y.append(row[-1])

    feature_names = np.array(X[0])

    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names


X, y, feature_names = load_dataset(sys.argv[1])
X, y = shuffle(X, y, random_state=7)

num_training = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_impurity_decrease=1)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\n### Random Forest regressor perfomance ###")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)

