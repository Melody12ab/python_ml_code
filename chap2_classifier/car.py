import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

input_file = 'car.data.txt'
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        # 最后一个字符是换行符
        data = line[:-1].split(',')
        X.append(data)

X = np.array(X)
# 将字符串转为数字
label_encoder = []
# empty 类似于zero，不过不要求全都为零，所以快很多，使用时请注意
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    # 每一个LabelEncoder用来Label列数据，转回使用inverse_transform
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
Y = X_encoded[:, -1].astype(int)

params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, Y)

accuracy = cross_val_score(classifier, X, Y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + str(round(100 * accuracy.mean(), 2)) + "%")

input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    # 再次使用不用再fit input_data需要是一个sequence，保持和fit时候一致的数据结构
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))

input_data_encoded = np.array(input_data_encoded)
# input_data_encoded需要是一个sequence，保持和fit时候一致的数据结构
output_class = classifier.predict([input_data_encoded])
print("Output class:", label_encoder[-1].inverse_transform(output_class))

############################
# Validation curves hyperparameters
from sklearn.model_selection import validation_curve

classifier = RandomForestClassifier(max_depth=4, random_state=7)
parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, Y, param_name="n_estimators",
                                                   param_range=parameter_grid, cv=5)
print("\n##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)

# Plot the curve
import matplotlib.pyplot as plt

plt.figure()
# average 中axis=1表示求每一行的均值
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title("Training curve")
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.show()

classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, Y, param_name="max_depth",
                                                   param_range=parameter_grid, cv=5)
print("\nParam: max_depth\nTraining scores:\n", train_scores)
print("\nParam: max_depth\nValidation scores:\n", validation_scores)
plt.figure()
# average 中axis=1表示求每一行的均值
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title("Validation curve")
plt.xlabel("Maximum depth of the tree")
plt.ylabel("Accuracy")
plt.show()

# Learning curves
from sklearn.model_selection import learning_curve

classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])
train_sizes, train_scores, validation_scores = learning_curve(classifier, X, Y, train_sizes=parameter_grid, cv=5)
print("\n##### LEARNING CURVES #####")
print("\nTraining scores:\n", train_scores)
print("\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1),
         color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
