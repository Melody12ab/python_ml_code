from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
import numpy as np

input_file = 'adult.data.txt'

X = []
Y = []

count_lessthan50k = 0
count_morethan50k = 0

num_images_threshold = 10000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k = count_lessthan50k + 1
        elif data[-1] == '>50' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k = count_morethan50k + 1

        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break

X = np.array(X)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
Y = X_encoded[:, -1].astype(int)

# classifier_gaussiannb=GaussianNB()
# classifier_gaussiannb.fit(X,Y)

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)
f1 = cross_val_score(classifier_gaussiannb, X, Y, scoring="f1_weighted", cv=5)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Testing encoding on single data instance
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13',
              'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male',
              '2174', '0', '40', 'United-States']
count = 0
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        # print(label_encoder[count].classes_)
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1

input_data_encoded = np.array(input_data_encoded)
print(input_data_encoded)
# Predict and print output for a particular datapoint
output_class = classifier_gaussiannb.predict([input_data_encoded])
# 返回给定值的label list
print(label_encoder[-1].inverse_transform(output_class)[0])
