import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB


# plot the classfifier graph boundaries
def plot_classifier(classifier, X, y):
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.show()


input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

classifier_guassionnb = GaussianNB()
classifier_guassionnb.fit(X, y)
y_pred = classifier_guassionnb.predict(X)

# compute accuracy
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 3), "%")

# plot data and boundaries
plot_classifier(classifier_guassionnb, X, y)

# crossvalidation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
classifier_guassionnb_new = GaussianNB()
classifier_guassionnb_new.fit(X_train, y_train)
y_test_pred = classifier_guassionnb_new.predict(X_test)
accuracy = 100 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 3), "%")

plot_classifier(classifier_guassionnb_new, X_test, y_test)

# accuracy: 模型的准确率
# precision:模型对负样本的区分能力，precision越高，说明模型对负样本的区分能力越强
# recall:分类模型对正样本的识别能力，recall 越高，说明模型对正样本的识别能力越强
# F1-score:两者的综合。F1-score 越高，说明分类模型越稳健
num_validation = 5
accuracy = cross_val_score(classifier_guassionnb, X, y, scoring="accuracy", cv=num_validation)
print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")

f1 = cross_val_score(classifier_guassionnb, X, y, scoring='f1_weighted', cv=num_validation)
print("F1: " + str(round(100 * f1.mean(), 2)) + "%")

precision = cross_val_score(classifier_guassionnb, X, y, scoring="precision_weighted", cv=num_validation)
print("Precision: " + str(round(100 * precision.mean(), 2)) + "%")

recall = cross_val_score(classifier_guassionnb, X, y, scoring="recall_weighted", cv=num_validation)
print("Recall: " + str(round(100 * recall.mean(), 2)) + "%")
