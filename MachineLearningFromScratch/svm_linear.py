from load_data import dt_train, dt_train_target, dt_test, dt_test_target, data_index
import svm_func as svm
import numpy as np

wt_lst=[
0.20717319464152695,
0.6206984801546556,
0.43006365979581385,
0.4580060620231049,
0.33883164044002245,
0.36739188157541647,
0.5832498372259184,
0.9209212798422121,
0.6737554996556483,
0.5508969453938781,
0.21220708634760788,
0.43266411739534283,]
wt_arr = np.array(wt_lst)


error=0;
for index in range(len(dt_train)):
    r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]
    if r<0:
        error+=1
print('error (initial weight):', error/len(dt_train)*100, '%\n')

C=10e12
wt_arr = svm.SVM(dt_train, dt_train_target, wt_arr, kernel='linear', C=10e12, plot=True)

res = svm.SVM_test(dt_test, dt_test_target, wt_arr)
print('score: ', res*100, '%')