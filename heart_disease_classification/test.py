from load_data import dt_train, dt_train_target, dt_test, dt_test_target, data_index
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
#print('Initial weight: ', *wt_arr, sep='\n')
#print('data_train', *dt_train, sep='\n')


error=0;
for index in range(len(dt_train)):
    r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]
    if r<0:
        error+=1
print('error: ', error/len(dt_train)*100, '%\n')


#print(*dt_train,sep='\n')

C=10e6
stp=1
epoch_limit =1000
epoch=0
hinge_loss_all=0
while epoch <= epoch_limit:
    hinge_loss=[]
    hinge_loss_last = hinge_loss_all
    reg_val = 1 / C

    for index in range(len(dt_train)):

        r = np.dot(dt_train[index], wt_arr)*dt_train_target[index]

        #print(r)

        if r>=1:
            hinge_loss.append(0)
            wt_arr = wt_arr - stp*reg_val*wt_arr
        else:
            hinge_loss.append(1-r)
            wt_arr = wt_arr + stp*(dt_train_target[index]*np.array(dt_train[index])-reg_val*wt_arr)

        hinge_loss_all = sum(hinge_loss)

    print('epoch:', epoch, ',hinge_loss:', round(hinge_loss_all,5), ',step: ', stp)

    if abs(hinge_loss_last - hinge_loss_all)<=0.1:
        stp = stp * 0.01

    if stp<=1e-300:
        break

    epoch+=1


print('length: \n')
error=0;
for index in range(len(dt_train)):
    reg_val = 1 / C
    r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]
    if r<0:
        error+=1
    print(r)
print('train data length: ', len(dt_train))
print('bias: ', error/len(dt_train)*100, '%\n')



error=0;
for index in range(len(dt_test)):
    reg_val = 1 / C
    r = np.dot(dt_test[index], wt_arr) * dt_test_target[index]
    if r<0:
        error+=1
print('test data length: ',len(dt_test))
print('score: ', (1-error/len(dt_test))*100, '%\n')




