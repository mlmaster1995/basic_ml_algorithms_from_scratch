import data_process as dp
from Logistic_Regression import Logistic_Regression
from sklearn.linear_model import LogisticRegression

# load data
data_raw, data_index = dp.load_data('heart_disease_data.csv')
data_rf = data_raw.copy().iloc[:, :-1].values.tolist()
data_rf_target = data_raw.copy().iloc[:, -1].values.tolist()
data_rf_normalized = dp.normalize_data(data_rf)

# get train and test data
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(data_rf_normalized, data_rf_target, split_ratio=0.7, seed=2)

# logistic regression test
# tunning the epoch to converge -> adjust reg_lambda to increase more
lr_clf = Logistic_Regression(data_train, data_train_target)
lr_clf.train(threshold=0.5, reg_lambda=0.1, stp=0.1, max_epoch=100, plot=False)
acc = lr_clf.score(data_test, data_test_target, threshold=0.5)
print(f'test acc: {acc * 100}%')

# scikit learn logistic regression
clf = LogisticRegression(random_state=1, solver='liblinear', multi_class='auto').fit(data_train, data_train_target)
acc = clf.score(data_test, data_test_target)
print(f'sklearn acc:{acc * 100}%')
