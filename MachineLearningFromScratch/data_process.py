import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random


# load & split data into train & test set
def load_data(name):
    path = os.getcwd()
    data_path = os.path.join(path, name)
    data_raw = pd.read_csv(data_path)
    data_index = data_raw.keys()

    return data_raw, data_index


# plot histogram of each attribute
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


# split data_test from data_train, split_ration=0.7, 70% data for training, 30% of data for testing
def split_train_test_data(data_refine, data_refine_target, split_ratio, rand=True, seed=2):
    random.seed(seed)
    data_train = []
    data_train_target = []
    data_test = []
    data_test_target = []
    train_length = int(len(data_refine) * split_ratio)

    if rand:
        for index in range(train_length):
            pos = random.randint(0, len(data_refine) - 1)
            data_train.append(data_refine.pop(pos))
            data_train_target.append(data_refine_target.pop(pos))

        data_test = data_refine
        data_test_target = data_refine_target

    else:

        data_refine_dic = {}

        for index in range(len(data_refine_target)):
            if data_refine_target[index] not in data_refine_dic:
                data_refine_dic[data_refine_target[index]] = []
                data_refine_dic[data_refine_target[index]].append(data_refine[index])
            else:
                data_refine_dic[data_refine_target[index]].append(data_refine[index])

        for key in list(data_refine_dic.keys()):

            train_length = int(len(data_refine_dic[key]) * split_ratio)

            for index in range(train_length - 1):
                data_train.append(data_refine_dic[key].pop(0))
                data_train_target.append(data_refine_target.pop(0))

            for item in data_refine_dic[key]:
                data_test.append(item)
            data_test_target = data_refine_target

    return data_train, data_train_target, data_test, data_test_target


# shuffle the categarized data
def shuffle_data_ca(data_ca, data_ca_target):
    data_shf = []
    data_shf_target = []
    for i in range(len(data_ca)):
        loc = random.randint(0, len(data_ca) - 1)
        data_shf.append(data_ca.pop(loc))
        data_shf_target.append(data_ca_target.pop(loc))

    return data_shf, data_shf_target


# scale all data into [0,1]
def scale_data(dt_train):
    dt_new = dt_train
    col_max_min = [(np.min(col), np.max(col)) for col in list(zip(*dt_new))]
    for row_index in range(len(dt_new)):
        for col_index in range(len(dt_new[row_index])):
            col_min = col_max_min[col_index][0]
            col_max = col_max_min[col_index][1]
            dt_new[row_index][col_index] = (dt_new[row_index][col_index] - col_min) / (col_max - col_min)
    return dt_new


# normalize the dataset
def normalize_data(data_set, bit=5):
    normalize_dataset = []
    data_rf_t = np.array(data_set.copy()).T.tolist()
    for row in data_rf_t:
        max_row = max(row)
        min_row = min(row)
        normalize_dataset.append([round((value - min_row) / (max_row - min_row), bit) for value in row])
    normalize_dataset = np.array(normalize_dataset).T.tolist()

    return normalize_dataset


# put all together
def data_put_all(filename, split_ratio, rand=True):
    # take all features into data_rf and targets into data_rf_target
    data_raw, data_raw_index = load_data(filename)
    data_raw = data_raw.drop(columns=['chol', 'fbs'])
    data_index = data_raw.keys()

    data_lst = data_raw.values.tolist()
    data_rf = [dt[:-1] for dt in data_lst]
    data_rf_target = list(list(zip(*data_lst))[-1])

    data_rf = scale_data(data_rf)

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
    dt_train, dt_train_target = shuffle_data_ca(dt_train, dt_train_target)

    return dt_train, dt_train_target, dt_test, dt_test_target, data_index
