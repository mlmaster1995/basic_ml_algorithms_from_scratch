import random
import numpy as np

## generate random weight w0,w1,w2,w3.....
def generate_weight(num_attr, start, end):
    wt_array = []
    for i in range(num_attr):
        wt_array.append(random.uniform(start, end))

    return wt_array

def SVM( dt_train, dt_train_target, wt_arr, kernal='linear', C=1, stp=1, epoch_limit=1000, stp_limit=1e-300, show_bias=True, stp_show=False):

    if kernal == 'linear':
        epoch = 0
        hinge_loss_all = 0
        reg_val = 1 / C
        while epoch <= epoch_limit:
            hinge_loss = []
            hinge_loss_last = hinge_loss_all


            for index in range(len(dt_train)):

                r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]

                if r >= 1:
                    hinge_loss.append(0)
                    wt_arr = wt_arr - stp * reg_val * wt_arr
                else:
                    hinge_loss.append(1 - r)
                    wt_arr = wt_arr + stp * (dt_train_target[index] * np.array(dt_train[index]) - reg_val * wt_arr)

                hinge_loss_all = sum(hinge_loss)

            if stp_show:
                print('epoch:', epoch, ',hinge_loss:', round(hinge_loss_all, 5), ',step: ', stp)

            if abs(hinge_loss_last - hinge_loss_all) <= 0.1:
                stp = stp * 0.01

            if stp <= stp_limit:
                break

            epoch += 1

        if show_bias:
            error = 0
            if stp_show:
                print('vector length: ')
            for index in range(len(dt_train)):
                r = np.dot(dt_train[index], wt_arr) * dt_train_target[index]
                if r < 0:
                    error += 1
                if stp_show:
                    print(r)
            bias = error / len(dt_train) * 100
            print('data train length: ', len(dt_train))
            print('bias: ', bias, '%\n')

        return (wt_arr, bias/100, C)


def SVM_test(dt_test, dt_test_target, wt_arr, C):
    error = 0
    reg_val = 1 / C
    for index in range(len(dt_test)):

        r = np.dot(dt_test[index], wt_arr) * dt_test_target[index]
        if r < 0:
            error += 1
    # print('test data length: ', len(dt_test))
    # print('variance: ', error / len(dt_test) * 100, '%\n')

    score = 1- error / len(dt_test)

    return (score, len(dt_test))


