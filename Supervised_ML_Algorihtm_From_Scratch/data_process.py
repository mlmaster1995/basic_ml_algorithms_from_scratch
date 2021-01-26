
# Created By Chris Young in 2019
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random
from copy import deepcopy


# load & split data into train & test set
def load_data(name):
    path = os.getcwd()
    data_path = os.path.join(path, name)
    data_raw = pd.read_csv(data_path)
    data_index = data_raw.keys()

    return data_raw, data_index


# normalize the dataset
def normalize_data(data_set, bit=5):
    data_set_array = np.array(deepcopy(data_set))
    if data_set_array.shape[0] > 1:
        normalize_dataset = []
        data_rf_t = data_set_array.T
        for row in data_rf_t:
            max_row = np.max(row)
            min_row = np.min(row)
            normalize_dataset.append([round((value - min_row) / (max_row - min_row), bit) for value in row])
        normalize_dataset = np.array(normalize_dataset).T.tolist()
    else:
        max_row = np.max(data_set_array)
        min_row = np.min(data_set_array)
        normalize_dataset = ((data_set_array - min_row) / (max_row - min_row)).tolist()

    return normalize_dataset


# merge dataset and classification target
def merge_dataset(dataset, dataset_target):
    new_dataset = []
    for item in zip(dataset, dataset_target):
        item[0].append(item[1])
        new_dataset.append(item[0])
    return new_dataset


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
def split_train_test_data(data_refine, data_refine_target, split_ratio=0.7, seed=None):
    if seed is not None:
        random.seed(seed)
    data_train = []
    data_train_target = []
    train_length = int(len(data_refine) * split_ratio)

    for index in range(train_length):
        pos = random.randint(0, len(data_refine) - 1)
        data_train.append(data_refine.pop(pos))
        data_train_target.append(data_refine_target.pop(pos))

    data_test = data_refine
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
