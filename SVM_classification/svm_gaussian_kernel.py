import data_process as dp
from svm_kernel_weights import wt_arr
import svm_func as svm
import numpy as np


score_lst=[]
for i in range(1,100):
    # load raw data and show correlation with target column
    data_raw, data_raw_index= dp.load_data('heart_disease_data.csv')


    # drop 'chol', 'fbs' du to low correlation to the target
    data_raw = data_raw.drop(columns=['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg',])
    data_raw_index = data_raw.keys()


    # covert from dataset dataframe to list
    data_lst = data_raw.values.tolist()


    # refine the dataset by removing the target columns
    data_rf = [dt[:-1] for dt in data_lst]
    data_rf_target = list(list(zip(*data_lst))[-1])


    # normalize the every columns value to [0,1]
    data_rf = dp.scale_data(data_rf)


    # modify dataset target to '1' or '-1'
    for index in range(len(data_rf_target)):
        if data_rf_target[index] == 0.0:
            data_rf_target[index] = -1


    # convert data_raw_index to a list so as to match the dataset columns
    data_index = data_raw_index.values.tolist()
    data_index = data_index[:-1]


    # split the data_rf into train & test setssvm_kernel_weights
    split_ratio= 0.8
    dt_train, dt_train_target, dt_test, dt_test_target = \
        dp.split_train_test_data(data_rf, data_rf_target, split_ratio, rand=True)


    # train the model
    wt_arr, C, stp = svm.SVM(dt_train, dt_train_target, wt_arr, kernel='gaussian', stp=0.1, C=10e10, sigma=0.8, plot=False)


    # test the model
    lama = np.array(dt_train)
    features_test = svm.gaussian_feature_make(dt_test, lama, sigma=0.8)
    res = svm.SVM_gaussian_test(features_test, dt_test_target, wt_arr, True, stp=0.01, C=10e10)
    # print('score: ', round(res*100, 3), '%')

    score_lst.append(round(res*100, 3))

print('max:', np.max(score_lst), '%, ', 'min:', np.min(score_lst), '%')
