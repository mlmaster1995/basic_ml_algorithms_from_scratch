import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random


## load & split data into train & test set
def load_data(name):
    path = os.getcwd()
    data_path = os.path.join(path, name)
    data_raw = pd.read_csv(data_path)
    data_raw = data_raw.drop(columns=['chol', 'fbs'])
    data_index = data_raw.keys()

    #print(data_index)
    return (data_raw, data_index)


## plot histogram of each attribute
def plot_hist(data_refine, data_refine_index):
    data_zip = list(zip(*data_refine))

    for index in range(len(data_zip)):
        each_attr = data_zip[index]
        low_b = math.floor(min(each_attr))
        upp_b = math.ceil(max(each_attr))
        plt.hist(each_attr, range=[low_b, upp_b])
        plt.title(data_refine_index[index], loc='center')
        plt.grid()
        plt.show()


## split data_test from data_train, split_ration=0.7, 70% data for training, 30% of data for testing
def split_train_test_data(data_refine, data_refine_target, split_ratio, rand=True):
    data_train = []
    data_train_target = []
    data_test=[]
    data_test_target=[]
    train_length = int(len(data_refine) * split_ratio)

    if rand:
        for index in range(train_length):
            pos = random.randint(0, len(data_refine) - 1)
            data_train.append(data_refine.pop(pos))
            data_train_target.append(data_refine_target.pop(pos))

        data_test = data_refine
        data_test_target = data_refine_target

    else:

        data_refine_dic={}

        for index in range(len(data_refine_target)):
            if data_refine_target[index] not in data_refine_dic:
                data_refine_dic[data_refine_target[index]]=[]
                data_refine_dic[data_refine_target[index]].append(data_refine[index])
            else:
                data_refine_dic[data_refine_target[index]].append(data_refine[index])

        for key in list(data_refine_dic.keys()):

            train_length = int(len(data_refine_dic[key]) * split_ratio)

            for index in range(train_length-1):
                data_train.append(data_refine_dic[key].pop(0))
                data_train_target.append(data_refine_target.pop(0))


            for item in data_refine_dic[key]:
                data_test.append(item)
            data_test_target = data_refine_target

    return (data_train, data_train_target, data_test, data_test_target)


# shuffle the categarized data
def shuffle_data_ca(data_ca, data_ca_target):
    data_shf = []
    data_shf_target = []
    for i in range(len(data_ca)):
        loc = random.randint(0, len(data_ca) - 1)
        data_shf.append(data_ca.pop(loc))
        data_shf_target.append(data_ca_target.pop(loc))

    return (data_shf, data_shf_target)

# put all together

def data_put_all(filename, split_ratio, rand=True):

    # take all features into data_rf and targets into data_rf_target
    data_raw, data_raw_index = load_data(filename)

    data_lst = data_raw.values.tolist()
    data_rf = [dt[:-1] for dt in data_lst]
    data_rf_target = list(list(zip(*data_lst))[-1])

    # normalize the age, trestbps, chol, thalach,
    for val in data_rf:
        val[0] = val[0] / 100
        val[2] = val[2] /10
        val[3] = val[3] / 1000
        val[5] = val[5] /1000
        val[6] = val[6] / 1000
        val[7] = val[7] / 10
        val[8] = val[8] /10
        val[9] = val[9] /10
        val[10] = val[10] /10

    # normalize feature input as x0, x1, x2, x3....1 for dot product
    for dt in data_rf:
        dt.append(1)

    # normalize feature input target as 1,-1
    for index in range(len(data_rf_target)):
        if data_rf_target[index] == 0.0:
            data_rf_target[index] = -1

    # get data index
    data_index = data_raw_index.values.tolist()
    data_index = data_index[:-1]

    dt_train, dt_train_target, dt_test, dt_test_target = \
        split_train_test_data(data_rf, data_rf_target, split_ratio, rand)

    # shuffle all data
    #if rand:
    dt_train, dt_train_target = shuffle_data_ca(dt_train, dt_train_target);


    return (dt_train, dt_train_target, dt_test, dt_test_target, data_index)























