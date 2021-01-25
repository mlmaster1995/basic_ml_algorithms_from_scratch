import data_process as dp
from sklearn.naive_bayes import GaussianNB
from Naive_Bays import NaiveBays

# load data
data_raw, data_index = dp.load_data('diabetes_data.csv')
common = 1

# test Naive Bays
data_raw_copy = data_raw.copy()
data_rf = data_raw.iloc[:, :-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
normalize_dataset = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(normalize_dataset, data_rf_target, split_ratio=0.8, seed=common)

evaluation = NaiveBays(data_train, data_train_target).evaluate(data_test, data_test_target)
print(f'NaiveBays acc: {round(evaluation * 100, 5)}%')

# Sklearn Naive Bays
data_rf = data_raw.iloc[:, :-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
normalize_dataset = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(normalize_dataset, data_rf_target, split_ratio=0.8, seed=common)

clf = GaussianNB()
clf.fit(data_train, data_train_target)
score2 = clf.score(data_test, data_test_target)
print(f'Scikit-Learn acc:{round(score2 * 100, 5)}%')
