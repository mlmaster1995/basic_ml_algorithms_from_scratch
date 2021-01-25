import data_process as dps
from Knn_classifier import KNN
import math
from sklearn.neighbors import KNeighborsClassifier

data_raw, data_index= dps.load_data('heart_disease_data.csv')
data_raw_copy = data_raw.copy()
data_target = data_raw['target'].values.tolist()
data_body = data_raw.iloc[:,:-1].values.tolist()
data_train, data_train_target, data_test, data_test_target = dps.split_train_test_data(data_body, data_target, split_ratio=0.7, seed=2)

# build knn classifier
knn = KNN(data_train, data_train_target)

# test KNN
K = knn.optimize_K(data_test, data_test_target, showtime=False)
# score = knn.score(data_test, data_test_target, K, parallel=True)
print(f'knn score:{K[0]*100}%')

# sklearn knn
neigh = KNeighborsClassifier(n_neighbors=K[1])
neigh.fit(data_train, data_train_target)
score = neigh.score(data_test, data_test_target)
print(f'scikit-learn score:{score*100}%')



