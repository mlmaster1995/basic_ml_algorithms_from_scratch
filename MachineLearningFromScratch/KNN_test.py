import data_process as dps
from KNN import KNN
from sklearn.neighbors import KNeighborsClassifier

# load dataset
data_raw, data_index= dps.load_data('heart_disease_data.csv')
data_raw_copy = data_raw.copy()
data_target = data_raw['target'].values.tolist()
data_body = data_raw.iloc[:,:-1].values.tolist()
data_train, data_train_target, data_test, data_test_target = dps.split_train_test_data(data_body, data_target, split_ratio=0.7, seed=3)

# build KNN classifier
knn = KNN(data_train, data_train_target)

# test KNN with optimized K
K = knn.optimize_K(data_test, data_test_target, showtime=False)
# evaluate = knn.evaulate(data_test, data_test_target, K[1], parallel=True)
# print(f'knn evaulate acc:{evaluate[0]*100}%')
print(f'KNN evaulate acc:{K[0]*100}%')

# sklearn KNN with optimized K
neigh = KNeighborsClassifier(n_neighbors=K[1])
neigh.fit(data_train, data_train_target)
acc = neigh.score(data_test, data_test_target)
print(f'Scikit-Learn KNN evaulate acc:{acc*100}%')



