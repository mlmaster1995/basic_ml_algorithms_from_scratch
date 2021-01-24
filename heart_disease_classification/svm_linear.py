from load_data import dt_train, dt_train_target, dt_test, dt_test_target, data_index
from svm_func import SVM, SVM_test
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
print('Initial weight: ', *wt_arr, sep='\n')
#print('data_train', *dt_train, sep='\n')


error=0;
for index in range(len(dt_train)):
    r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]
    if r<0:
        error+=1
print('error (initial weight): ', error/len(dt_train)*100, '%\n')

wt_arr, bias, C = SVM(dt_train, dt_train_target, wt_arr, kernal='linear', C=10e12, stp=1, epoch_limit=1000,
                      stp_limit=1e-300, show_bias=True, stp_show=False)

res = SVM_test(dt_test, dt_test_target, wt_arr, C)
print('data tst length: ', res[1])
print('score: ', res[0]*100, '%')