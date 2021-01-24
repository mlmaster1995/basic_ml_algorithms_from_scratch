import data_process as dp
from logistic_regression_classifier import Logistic_Regression
from sklearn.linear_model import LogisticRegression
from logistic_regression_func import generate_weight, sigmoid_sgv, cost_func, weight_update, calc_error_rate
import numpy as np
import matplotlib.pyplot as plt


data_raw, data_index = dp.load_data('heart_disease_data.csv')
data_rf = data_raw.copy().iloc[:,:-1].values.tolist()
data_rf_target = data_raw.copy().iloc[:,-1].values.tolist()
data_rf_normalized = dp.normalize_data(data_rf)

data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(data_rf_normalized, data_rf_target, split_ratio=0.7, seed=2)

'''
# Logistic Regression
wt_arr = generate_weight(len(data_train[0]), val_range=[0, 1], seed=2)
print(f'Initial weight:{wt_arr}')

threshold = 0.5
reg_lambda= 0
stp= 0.1
epoch= 1000

resolution = 3
stop_limit = 1e-300
stp_limit = 1/pow(10, resolution)

ep=0
epoch_limit=[]
cost_lst=[]


# train the model
while ep < epoch:
    # setup error default
    er = 0
    epoch_limit.append(ep)

    # calc train data hypothesis value
    h_x_lst = [sigmoid_sgv(np.dot(wt_arr, np.array(each_dt))) for each_dt in data_train]

    # calculate error
    cost= calc_error_rate(wt_arr, data_train, data_train_target, threshold)
    # cost = cost_func(h_x_lst, data_train_target, threshold)
    cost_lst.append(cost)

    # adjust step size
    # if ep > 0:
    #     if abs(round(cost_lst[-2], 4) - round(cost_lst[-1], 4)) < stp_limit:
    #         stp = stp * stp_limit
    #     if stp < stop_limit:
    #         break

    # update weight values
    h_x_arr = np.array(h_x_lst)
    y_arr = np.array(data_train_target)
    x_arr = np.array(data_train)

    if reg_lambda > len(x_arr):
        reg_lambda = x_arr

    wt_arr = weight_update(wt_arr, h_x_arr, y_arr, x_arr, stp, reg_lambda, threshold)
    ep += 1



# clculate model training error rate
error_rate = calc_error_rate(wt_arr, data_train, data_train_target, threshold)
print('bias:', error_rate * 100, '%')

# calculate model testing error
test_error = calc_error_rate(wt_arr, data_test, data_test_target, threshold)
print(f'variance: {test_error*100}%')
print(f'score: {(1-test_error)*100}%')

# print final weight
print('Final weight: \n', wt_arr, '\n')

# plot the error rate
plt.plot(epoch_limit, cost_lst)
plt.axis([0, len(epoch_limit), 0, 1])
plt.title('Error Vs Epoch')
plt.xlabel('epoch')
plt.ylabel('Error')
plt.grid()
plt.show()
'''

# logistic regression test
# tunning the epoch to converge -> adjust reg_lambda to increase more
lr_clf = Logistic_Regression(data_train, data_train_target)
lr_clf.train(threshold=0.5, reg_lambda=0.1, stp=0.1, max_epoch=50, plot = False)
score = lr_clf.score(data_test, data_test_target, threshold=0.5)
print(f'\nModel runtime: {round(lr_clf.training_time,3)}seconds')
print(f'training score: {(1-lr_clf.model_bias)*100}%')
print(f'test score: {score*100}%')


# scikit learn logistic regression
clf = LogisticRegression(random_state=1, solver='liblinear', multi_class='auto').fit(data_train, data_train_target)
score = clf.score(data_test, data_test_target)
print(f'\nsklearn score:{score*100}%')








