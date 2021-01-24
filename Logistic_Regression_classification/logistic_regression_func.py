import random
import numpy as np
import math
import statistics as st

# generate random weight for attribute and return an array
def generate_weight(num_attr, val_range=[0, 0.5], seed=2):
    random.seed(seed)
    wt_array=[]
    for i in range(num_attr):
        wt_array.append(random.uniform(val_range[0], val_range[1]))

    return np.array(wt_array)

# calculate sigmoid value
def sigmoid_sgv(z):
    return 1/(1+math.exp(-z))

# define cost function
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
    # print('***',mean_cost)

    return mean_cost

# define weight update function
def weight_update(wt_arr, h_x_arr, y_arr, x_arr, stp, reg_lambda=0, threshold=0.5):


    er=0
    for val in list(zip(h_x_arr, y_arr, x_arr)):

        if round(val[0], 3) >= threshold and val[1] == 1:
            continue
        elif round(val[0], 3) < threshold and val[1] == 0:
            continue
        else:
            er += (val[0] - val[1]) / len(x_arr)
            wt_upt = er * val[2]
            wt_arr = wt_arr * (1 - stp * reg_lambda / len(x_arr)) - stp * wt_upt

            # wt_arr = wt_arr - stp*wt_upt


    # wt_arr = wt_arr*(1-(stp*reg_lambda)/len(x_arr))-stp*sum(h_x_arr-y_arr)/len(x_arr)

    return wt_arr

# calculate the error rate for the traning set in the model
def calc_error_rate(wt_arr, dt_train, dt_train_target, threshold):
    #print(*list(zip(h_x_lst,dt_train_target)),sep='\n')

    h_x_lst = [sigmoid_sgv(np.dot(wt_arr, np.array(each_dt))) for each_dt in dt_train]
    error=0
    for item in zip(h_x_lst, dt_train_target):
        if round(item[0], 3) >= threshold and item[1] != 1:
            error += 1
        elif round(item[0], 3) < threshold and item[1] != 0:
            error += 1

    return error/len(dt_train_target)