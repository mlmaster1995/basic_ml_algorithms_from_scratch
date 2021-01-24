import data_process as dp
from sklearn.naive_bayes import GaussianNB
from naive_bayes_func import calc_prob_likelihood_evidence, feature_mean_devi, printdict, calc_prior
import numpy as np
from naive_bays_classifier import NaiveBays

data_raw, data_index= dp.load_data('diabetes_data.csv')
common=1

# test naive bays
data_raw_copy = data_raw.copy()
data_rf = data_raw.iloc[:, :-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
normalize_dataset = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(normalize_dataset, data_rf_target, split_ratio=0.8, seed= common)

nb_clf = NaiveBays(data_train, data_train_target)
score = nb_clf.score(data_test, data_test_target)
print(f'me score: {round(score*100, 5)}%')


# sklearn naive bay score
data_rf = data_raw.iloc[:, :-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
normalize_dataset = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(normalize_dataset, data_rf_target, split_ratio=0.8, seed= common)

clf = GaussianNB()
clf.fit(data_train, data_train_target)
score2 = clf.score(data_test, data_test_target)
print(f'sk learn score:{round(score2*100,5)}%')



























