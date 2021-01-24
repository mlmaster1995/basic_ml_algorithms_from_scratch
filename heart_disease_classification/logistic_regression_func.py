import math
import statistics as st
import random
import numpy as np
import matplotlib.pyplot as plt


## define sigmoid function
def sigmoid_lst(z_list):
    return [1 / (1 + math.exp(-value)) for value in z_list]


def sigmoid_sgv(z):
    return 1 / (1 + math.exp(-z))


## define cost function
def cost_func(h_x_lst, y_lst, threshold):
    cost = [];
    for index in range(len(h_x_lst)):
        if h_x_lst[index] <= 0 or (1 - h_x_lst[index]) <= 0:
            print('\n hx out of range')
            print(*h_x_lst, sep='\n')
            break

        if round(h_x_lst[index], 3) >= threshold and y_lst[index] == 1:
            # print('*',h_x_lst[index],y_lst[index])
            cost.append(0)
        elif round(h_x_lst[index], 3) < threshold and y_lst[index] == 0:
            # print('**',h_x_lst[index],y_lst[index])
            cost.append(0)
        else:
            # print('***',h_x_lst[index],y_lst[index])
            cal = (y_lst[index] * math.log(h_x_lst[index]) + (1 - y_lst[index]) * math.log(1 - h_x_lst[index]))
            # print(cal)
            cost.append(cal)

    mean_cost = abs(st.mean(cost))
    print('***', mean_cost)

    return st.mean(cost)


## define weight update function
def weight_update(wt_arr, h_x_arr, y_arr, x_arr, stp, reg_lambda=0, threshold=0.5):
    for val in list(zip(h_x_arr, y_arr, x_arr)):

        if round(val[0], 3) >= threshold and val[1] == 1:
            continue
        elif round(val[0], 3) < threshold and val[1] == 0:
            continue
        else:
            er = (val[0] - val[1]) / len(x_arr)
            wt_upt = er * val[2]
            wt_arr = wt_arr * (1 - stp * reg_lambda / len(x_arr)) - stp * wt_upt

    return wt_arr


## generate random weight for attribute and return an array
def generate_weight(num_attr):
    wt_array = []
    for i in range(num_attr):
        wt_array.append(random.uniform(0, 0.5))

    return np.array(wt_array)


## calculate the error rate for the traning set in the model
def calc_error_rate(h_x_lst, dt_train_target, threshold=0.5):
    # print(*list(zip(h_x_lst,dt_train_target)),sep='\n')
    error = 0;
    for item in zip(h_x_lst, dt_train_target):
        if round(item[0], 3) >= threshold and item[1] != 1:
            error += 1
        elif round(item[0], 3) < threshold and item[1] != 0:
            error += 1
    # print('Error Number: ',error)
    return error / len(dt_train_target)


## logistic regression function
def logistic_regression(epoch, epoch_plot, wt_arr, dt_train, dt_train_target, threshold, reg_lambda, plot=True):
    cost_lst = []

    # train the model
    while epoch > 0:
        # setup error default
        er = 0

        # calc train data hypothesis value
        h_x_lst = [sigmoid_sgv(np.dot(wt_arr, np.array(each_dt)) / scale) for each_dt in dt_train]

        # calculate error
        cost = calc_error_rate(h_x_lst, dt_train_target, threshold)
        cost_lst.append(cost)

        # update weight values
        h_x_arr = np.array(h_x_lst)
        y_arr = np.array(dt_train_target)
        x_arr = np.array(dt_train)
        wt_arr = weight_update(wt_arr, h_x_arr, y_arr, x_arr, stp, reg_lambda, threshold)

        epoch -= 1

    # clculate model training error rate
    bias = calc_error_rate(h_x_lst, dt_train_target, threshold)

    # plot the error rate
    if plot:
        plt.plot(list(range(epoch_plot)), cost_lst)
        plt.axis([0, epoch_plot, 0, 1])
        plt.title('Error Vs Epoch')
        plt.xlabel('epoch')
        plt.ylabel('error rate')
        plt.grid()
        plt.show()

    return (wt_arr, bias)