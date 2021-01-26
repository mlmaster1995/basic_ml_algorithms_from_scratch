import data_process as dp
from SVM import SVM_Kernel, SVM_Linear

# load data
data_raw, data_index = dp.load_data('heart_disease_data.csv')
data_rf = data_raw.iloc[:, 0:-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
data_rf = dp.normalize_data(data_rf)
data_train, data_train_target, data_test, data_test_target = dp.split_train_test_data(data_rf, data_rf_target, 0.7, seed=2)


# gaussian kernel
svm_gaussian = SVM_Kernel(data_train, data_train_target, 'gaussian')
svm_gaussian.train(C=1e10, sigma=0.8, degree=2, bias=0, stp=1, epoch_limit=1000, stp_limit=1e-100, plot=False, stp_show=False)
score_bias = svm_gaussian.score(data_test, data_test_target)
print(f'gaussian bias:{round((1-score_bias)*100,3)}%')
score = svm_gaussian.score(data_test, data_test_target)
print(f'gaussian test score:{round(score*100,3)}%')


print('*'*100)
# polynomial kernel
svm_poly = SVM_Kernel(data_train, data_train_target, 'polynomial')
svm_poly.train(C=1, degree=2, bias=1, stp=1, epoch_limit=1000, stp_limit=1e-100, stp_show=False, plot=False)
score_bias = svm_poly.score(data_train, data_train_target)
print(f'poly bias:{round((1-score_bias)*100,3)}%')
score = svm_poly.score(data_test, data_test_target)
print(f'poly test score:{round(score*100,3)}%')


print('*'*100)
# linear kernel
svm_linear = SVM_Linear(data_train, data_train_target)
svm_linear.train(C=1e6, stp=1, epoch_limit=1000, stp_limit=1e-100, stp_show=False, plot=False)
score_bias = svm_linear.score(data_train, data_train_target)
print(f'linear bias:{round((1-score_bias)*100,3)}%')
score = svm_linear.score(data_test, data_test_target)
print(f'linear test score:{round(score*100,3)}%')
