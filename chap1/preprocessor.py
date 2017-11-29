import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, -5.4],
    [0, 4, -0.3, -2.1],
    [1, 3.3, -1.9, -4.3]
])

# mean removal 把每个特征值的均值移除，保证特征值均值为0 可以消除特征彼此之间的偏差(bias)
data_standardized = preprocessing.scale(data)
print(data_standardized)
print("\nMean =", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))

# scaling 范围缩放  每个特征的述职范围可能变化很大，将特征范围缩放到合理的大小
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\n Min max scaled data =\n", data_scaled)

# normalization 归一化  对特征向量的值进行调整，保证每个特征向量的值都缩放到相同的数值范围
# 确保数据点没有因为特征的基本性质产生较大差异，确保数据处于同一数量级
data_normalized = preprocessing.normalize(data, norm='l1')
print("\n L1 normalized data =", data_normalized)

# binarization 将数值特征向量转为布尔类型向量
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\n Binarized data =", data_binarized)

# One-hot-encoding 处理稀疏、散乱分布的数据 用于收紧特征向量
encoder = preprocessing.OneHotEncoder()
encoder.fit([
    [0, 2, 1, 12],
    [1, 3, 5, 3],
    [2, 3, 2, 12],
    [1, 2, 4, 3]
])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\n Encoded vector =", encoded_vector)
