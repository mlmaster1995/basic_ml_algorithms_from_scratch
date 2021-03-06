# Created By Chris Young in 2019
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from copy import deepcopy


class Logistic_Regression:
    def __init__(self, dataset, dataset_target):
        # dataset storage
        self.__dataset = deepcopy(dataset)
        self.__dataset_target = deepcopy(dataset_target)
        self.training_time = None

        # params for initial weight
        self.weight_seed = 2
        self.weight_range = [0, 1]
        self.initial_weight = None
        self.trained_weight = None

        # decent gradient step params
        self.converted_epoch = None
        self.__epoch_plt = []
        self.__cost_plt = []
        self.threshold = 0.5

        # bias
        self.model_bias = None

        # round digit
        self.round_digit = 5

    # generate random weight for attribute and return an array
    def __generate_weight(self):
        random.seed(self.weight_seed)
        wt_lst = []
        for i in range(len(self.__dataset[0])):
            val = round(random.uniform(self.weight_range[0], self.weight_range[1]), self.round_digit)
            wt_lst.append(val)
        return wt_lst

    def generate_weight(self, index_len, val_range=[0, 1], seed=2):
        random.seed(seed)
        wt_lst = []
        for i in range(index_len):
            wt_lst.append(random.uniform(val_range[0], val_range[1]))
        return wt_lst

    # define a sigmoid function
    def __sigmoid_sgv(self, z):
        val = round(1 / (1 + math.exp(-z)), self.round_digit)
        return val

    # define weight update function
    def __weight_update(self, h_x_arr, y_arr, x_arr, stp, reg_lambda, threshold):
        er = 0
        for val in list(zip(h_x_arr, y_arr, x_arr)):
            if round(val[0], 8) >= threshold and val[1] == 1:
                continue
            elif round(val[0], 8) < threshold and val[1] == 0:
                continue
            else:
                er += (val[0] - val[1]) / len(x_arr)
                wt_upt = np.around(er * val[2], decimals=self.round_digit)
                # self. trained_weight = self.trained_weight * (1 - stp * reg_lambda / len(x_arr)) - stp * wt_upt
                val = np.around(reg_lambda * self.trained_weight / len(x_arr), decimals=self.round_digit)
                self.trained_weight = self.trained_weight - stp * (wt_upt - val)

    # calculate the error rate for the traning set in the model
    def __calc_error_rate(self, wt_arr, dt_train, dt_train_target, threshold):
        h_x_lst = [self.__sigmoid_sgv(np.dot(wt_arr, np.array(each_dt))) for each_dt in dt_train]
        error = 0
        for item in zip(h_x_lst, dt_train_target):
            if round(item[0], 8) >= threshold and item[1] != 1:
                error += 1
            elif round(item[0], 8) < threshold and item[1] != 0:
                error += 1
        return error / len(dt_train_target)

    # model
    def train(self, threshold, reg_lambda, stp, max_epoch, plot=False):
        startime = time.time()
        self.initial_weight = np.array(self.__generate_weight())
        self.trained_weight = self.initial_weight.copy()
        self.threshold = threshold

        ep = 0
        # train the model
        while ep < max_epoch:
            # save epoch for plotting
            self.__epoch_plt.append(ep)

            # calc train data hypothesis value
            h_x_lst = [self.__sigmoid_sgv(np.dot(self.trained_weight, np.array(each_dt))) for each_dt in self.__dataset]

            # calculate error and save for plotting
            cost = self.__calc_error_rate(self.trained_weight, self.__dataset, self.__dataset_target, threshold)
            self.__cost_plt.append(cost)

            # update weight values
            h_x_arr = np.array(h_x_lst)
            y_arr = np.array(self.__dataset_target)
            x_arr = np.array(self.__dataset)

            # update model weights
            self.__weight_update(h_x_arr, y_arr, x_arr, stp, reg_lambda, threshold)

            # update the epoch
            ep += 1

        # clculate model training error rate & update the converge epoch
        self.model_bias = self.__calc_error_rate(self.trained_weight, self.__dataset, self.__dataset_target, threshold)
        self.converted_epoch = len(self.__epoch_plt)

        # plot the error rate
        if plot:
            plt.plot(self.__epoch_plt, self.__cost_plt)
            plt.axis([0, len(self.__epoch_plt), 0, 1])
            plt.title('Bias')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.grid()
            plt.show()

        self.training_time = time.time() - startime
        return self

    def score(self, datatest, datatest_target, threshold):
        error = self.__calc_error_rate(self.trained_weight, datatest, datatest_target, threshold)
        return 1 - error

    def classify_ensemble(self, single_dataset):
        single_dataset = deepcopy(single_dataset)
        h_x = self.__sigmoid_sgv(np.dot(self.trained_weight, np.array(single_dataset)))
        res = 1 if h_x >= self.threshold else 0
        return res
