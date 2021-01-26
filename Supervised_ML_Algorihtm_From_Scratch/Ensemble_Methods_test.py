import data_process as dp
from Ensemble_Methods import EnsembleLearner
from Decision_Regression_Forest import RandomForest
from KNN import KNN
from Logistic_Regression import Logistic_Regression
from Naive_Bays import NaiveBays
from SVM import SVM_Kernel

# load data
data_raw, data_index = dp.load_data('heart_disease_data.csv')
data_rf = data_raw.iloc[:, 0:-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
data_rf = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = dp.split_train_test_data(data_rf, data_rf_target, 0.7,
                                                                                      seed=2)

# random froests classifier
forests_clf = RandomForest()
forests_clf.build_forests(data_train, data_train_target,
                          tree_number=200, max_features=10, random_subspaces=False, max_depth=None, parallel=True)
print(f'random forest score: {round(forests_clf.score(data_test, data_test_target) * 100, 3)}%')

# knn classifier
knn = KNN(data_train, data_train_target)
K = knn.optimize_K(data_test, data_test_target, showtime=False)
print(f'knn score:{round(K[0] * 100, 3)}%')

# logistic regression classifier
lr_clf = Logistic_Regression(data_train, data_train_target)
lr_clf.train(threshold=0.5, reg_lambda=0.1, stp=0.1, max_epoch=50, plot=False)
score = lr_clf.score(data_test, data_test_target, threshold=0.5)
print(f'logistic regression score: {round(score * 100, 3)}%')

# naive bays classifier
nb_clf = NaiveBays(data_train, data_train_target)
score = nb_clf.score(data_test, data_test_target)
print(f'naive bays score: {round(score * 100, 3)}%')

# svm classifier
svm_gaussian = SVM_Kernel(data_train, data_train_target, 'gaussian')
svm_gaussian.train(C=1e10, sigma=0.8, degree=2, bias=0, stp=1, epoch_limit=1000, stp_limit=1e-100, plot=False,
                   stp_show=False)
score_bias = svm_gaussian.score(data_test, data_test_target)
print(f'svm gaussian score:{round(score_bias * 100, 3)}%')

# ensemble learning classifier
ensemble_clf = EnsembleLearner(data_train, data_train_target)
score = ensemble_clf.hard_voting_score(data_test, data_test_target, forests_clf, knn, lr_clf, svm_gaussian)
print(f'## hard volting score:{round(score * 100, 3)}%')

# adaBoost classifier
num = 300
boost_clf = EnsembleLearner(data_train, data_train_target)
boost_clf.adaBoost_train(estimator_limit=num, learning_rate=1, base_estimator='decision tree')
score = boost_clf.adaBoost_score(data_test, data_test_target)
print(f'## adaboost classifier score:{round(score * 100, 3)}%')
