import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# gaussian & polynomial kernel SVM_Linear
class SVM_Kernel:
    def __init__(self, dataset, dataset_target, kernel=None):
        # dataet parameters
        self.dataset = deepcopy(dataset)
        self.dataset_target = self.__process_dataset_target(deepcopy(dataset_target))
        self.kernel = kernel

        # gaussian kernel parameters
        self.weight = None
        self.wrt_range = [0, 1]
        self.wrt_seed = 2
        self.lama = np.array(self.dataset)
        self.sigma = None

        # polynomial kernal parameters
        self.degree = 2
        self.bias = 0

        # store misclassification dataset
        self.misclassification = [[], []]

    def __process_dataset_target(self, dataset_target):
        for index in range(len(dataset_target)):
            if dataset_target[index] == 0:
                dataset_target[index] = -1
            else:
                continue
        return dataset_target

    # generate random weight w0,w1,w2,w3.....
    def generate_weight(self, num_attr, seed=2):
        random.seed(seed)
        wt_array = []
        for i in range(num_attr):
            wt_array.append(random.uniform(self.wrt_range[0], self.wrt_range[1]))
        return np.array(wt_array)

    # generate gaussian features
    def __gaussian_feature_make(self, dataset, lama, sigma, round_limit=5):
        feature = []
        for val in dataset:
            feature.append([round(self.__gaussian_kernel(val, sig, sigma), round_limit) for sig in lama])
            feature[-1].insert(0, 1)
        return feature

    # gaussian_kernel func
    def __gaussian_kernel(self, x_ipt, lama_single, sigma):
        prob = np.exp(-(np.linalg.norm(abs(x_ipt - lama_single))) ** 2 / (2 * sigma ** 2))
        return prob

    # polynomial_kernel func
    def __poly_kernel(self, x_ipt, lama_single, degree=2, bias=0):
        poly_val = (np.dot(x_ipt, lama_single) + bias) ** degree
        return poly_val

    # generate poly features
    def __poly_feature_make(self, dt_train, lama, degree=2, bias=0):
        feature = []
        for val in dt_train:
            feature.append([self.__poly_kernel(val, sig, degree, bias) for sig in lama])
            feature[-1].insert(0, 1)
        return feature

    # train SVM_Linear model
    def train(self, C=1, sigma=0.5, degree=2, bias=0, stp=1, epoch_limit=1000, stp_limit=1e-300, stp_show=True,
              plot=False):
        epoch = 0
        hinge_loss_all = 0
        reg_val = 1 / C
        self.sigma = sigma
        epoch_plt = []
        hinge_loss_plt = []

        if self.kernel == 'gaussian':
            features = self.__gaussian_feature_make(self.dataset, self.lama, self.sigma, round_limit=5)
            self.weight = self.generate_weight(len(features[0]), self.wrt_seed)
        elif self.kernel == 'polynomial':
            features = self.__poly_feature_make(self.dataset, self.lama, self.degree, self.bias)
            self.weight = self.generate_weight(len(features[0]), self.wrt_seed)
        else:
            raise NotImplementedError('Kernel cannot be empty...')
            return -1

        while epoch <= epoch_limit:
            hinge_loss = []
            epoch_plt.append(epoch)
            hinge_loss_last = hinge_loss_all

            for index in range(len(features)):
                r = np.dot(features[index], self.weight) * self.dataset_target[index]

                if r >= 1:
                    hinge_loss.append(0)
                    self.weight = self.weight - stp * reg_val * self.weight
                else:
                    hinge_loss.append(1 - r)
                    self.weight = self.weight + stp * (
                                self.dataset_target[index] * np.array(features[index]) - reg_val * self.weight)
                hinge_loss_all = sum(hinge_loss)

            hinge_loss_plt.append(hinge_loss_all)

            if abs(hinge_loss_last - hinge_loss_all) <= 0.1:
                stp = stp * 0.1
            if stp <= stp_limit:
                break
            epoch += 1

            if stp_show:
                print('epoch:', epoch, ',hinge_loss:', round(hinge_loss_all, 5), ',step: ', stp)

        if plot:
            # plot hinge loss
            plt.plot(epoch_plt, hinge_loss_plt)
            plt.title('hinge loss vs epochs')
            plt.xlabel('epochs')
            plt.ylabel('hingle loss')
            plt.grid()
            plt.show()

    # test SVM_Linear model
    def score(self, dataset, dataset_target):
        dataset_target = self.__process_dataset_target(deepcopy(dataset_target))

        if self.kernel is 'gaussian':
            features = self.__gaussian_feature_make(dataset, self.lama, self.sigma, round_limit=5)
        elif self.kernel is 'polynomial':
            features = self.__poly_feature_make(dataset, self.lama, self.degree, self.bias)

        error = 0
        for index in range(len(features)):
            r = np.dot(features[index], self.weight) * dataset_target[index]
            if r >= 0:
                continue
            else:
                self.misclassification[0].append(dataset[index])
                self.misclassification[1].append(dataset_target[index])
                error += 1

        error_rate = error / len(features)
        return (1 - error_rate)

    # classify newdataset
    def classify_ensemble(self, single_data):
        dataset = [deepcopy(single_data)]
        if self.kernel is 'gaussian':
            features = self.__gaussian_feature_make(dataset, self.lama, self.sigma, round_limit=5)
        elif self.kernel is 'polynomial':
            features = self.__poly_feature_make(dataset, self.lama, self.degree, self.bias)
        r = np.dot(features[0], self.weight)
        return 1 if r >= 0 else 0


# linear SVM_Linear
class SVM_Linear:
    def __init__(self, dataset, dataset_target):
        # dataet parameters
        self.dataset = self.__process_dataset(deepcopy(dataset))
        self.dataset_target = self.__process_dataset_target(deepcopy(dataset_target))
        self.kernel = 'linear'

        # weight parameters
        self.wrt_range = [0, 1]
        self.wrt_seed = 2
        self.weight = self.__generate_weight(len(self.dataset[0]), self.wrt_seed)

        # store misclassification dataset
        self.misclassification = [[], []]

    def __process_dataset(self, dataset):
        for index in range(len(dataset)):
            dataset[index].append(1)
        return dataset

    def __process_dataset_target(self, dataset_target):
        for index in range(len(dataset_target)):
            if dataset_target[index] == 0:
                dataset_target[index] = -1
            else:
                continue
        return dataset_target

    # generate random weight w0,w1,w2,w3.....
    def __generate_weight(self, num_attr, seed=2):
        random.seed(seed)
        wt_array = []
        for i in range(num_attr):
            wt_array.append(random.uniform(self.wrt_range[0], self.wrt_range[1]))
        return np.array(wt_array)

    # train SVM_Linear model
    def train(self, C=1, stp=1, epoch_limit=1000, stp_limit=1e-100, stp_show=True, plot=False):
        epoch = 0
        hinge_loss_all = 0
        reg_val = 1 / C
        epoch_plt = []
        hinge_loss_plt = []

        while epoch <= epoch_limit:
            hinge_loss = []
            epoch_plt.append(epoch)
            hinge_loss_last = hinge_loss_all

            for index in range(len(self.dataset)):
                r = np.dot(self.dataset[index], self.weight) * self.dataset_target[index]

                if r >= 1:
                    hinge_loss.append(0)
                    self.weight = self.weight - stp * reg_val * self.weight
                else:
                    hinge_loss.append(1 - r)
                    self.weight = self.weight + stp * (
                                self.dataset_target[index] * np.array(self.dataset[index]) - reg_val * self.weight)
                hinge_loss_all = sum(hinge_loss)

            hinge_loss_plt.append(hinge_loss_all)

            if abs(hinge_loss_last - hinge_loss_all) <= 0.1:
                stp = stp * 0.1
            if stp <= stp_limit:
                break
            epoch += 1

            if stp_show:
                print('epoch:', epoch, ',hinge_loss:', round(hinge_loss_all, 5), ',step: ', stp)

        if plot:
            # plot hinge loss
            plt.plot(epoch_plt, hinge_loss_plt)
            plt.title('hinge loss vs epochs')
            plt.xlabel('epochs')
            plt.ylabel('hingle loss')
            plt.grid()
            plt.show()

    # test SVM_Linear model
    def score(self, dataset, dataset_target):
        dataset_target = self.__process_dataset_target(deepcopy(dataset_target))
        dataset = self.__process_dataset(deepcopy(dataset))

        error = 0
        for index in range(len(dataset)):
            r = np.dot(dataset[index], self.weight) * dataset_target[index]
            if r >= 0:
                continue
            else:
                self.misclassification[0].append(dataset[index])
                self.misclassification[1].append(dataset_target[index])
                error += 1

        error_rate = error / len(dataset)
        return (1 - error_rate)

    # classify newdataset
    def classify(self, single_data):
        dataset = [deepcopy(single_data)]
        processed_dataset = self.__process_dataset(dataset)

        r = np.dot(processed_dataset, self.weight)
        return 1 if r >= 0 else 0

    # classify newdataset in emsemble learning
    def classify_ensemble(self, single_data):
        dataset = [deepcopy(single_data)]
        processed_dataset = self.__process_dataset(dataset)

        r = np.dot(processed_dataset, self.weight)
        return 1 if r >= 0 else 0
