from copy import deepcopy
import numpy as np

class LinearRegression:
    def __init__(self, dataset, dataset_target):
        self.dataset = deepcopy(dataset)
        self.dataset_target = deepcopy([dataset_target])

        self.dataset_arr = np.array(self.dataset)
        self.dataset_target_arr = np.array(self.dataset_target).T

        self.shift_length=1e-10
        self.parameter_arr = None # column

        self.test_list = []

    # def train_model_original(self):
    #     A_T = self.dataset_arr.T
    #     A = self.dataset_arr
    #
    #     A_sqr = np.dot(A_T, A)
    #     shift_size = A_sqr.shape[0]
    #     shift_matrix = np.identity(shift_size, dtype=float)*self.shift_length
    #     A_shifted = A_sqr - shift_matrix
    #
    #     A_inv = np.linalg.inv(A_shifted)
    #     A_inv_transpose = np.dot(A_inv, A_T)
    #     self.parameter_arr = np.dot(A_inv_transpose, self.dataset_target_arr)
    #
    #     # print('haha')

    def train_model(self, reg=0):
        A_T = self.dataset_arr.T
        A = self.dataset_arr

        A_sqr = np.dot(A_T, A)
        if reg == 0:
            shift_size = A_sqr.shape[0]
            shift_matrix = np.identity(shift_size, dtype=float) * self.shift_length
            A_shifted = A_sqr - shift_matrix
        else:
            shift_size = A_sqr.shape[0]
            shift_matrix = np.identity(shift_size, dtype=float) * reg
            A_shifted = A_sqr - shift_matrix

        A_inv = np.linalg.inv(A_shifted)
        A_inv_transpose = np.dot(A_inv, A_T)
        self.parameter_arr = np.dot(A_inv_transpose, self.dataset_target_arr)

        # print('haha')

    def calc_error(self, dataset_test, dataset_test_target):
        self.test_list = []
        error_list = []
        for index in range(len(dataset_test)):
            arr = np.array([dataset_test[index]]) # row
            res = np.dot(arr, self.parameter_arr)

            self.test_list.append((res, dataset_test_target[index]))
            error_list.append(abs(res-dataset_test_target[index]))
            # error_list.append((res-dataset_test_target[index])**2)

        error = sum(error_list)/len(dataset_test)

        # print('haha')

        return error[0][0]

    def score(self, dataset_test ,dataset_test_target):
        error = self.calc_error(dataset_test, dataset_test_target)
        score = 1-error

        return score


