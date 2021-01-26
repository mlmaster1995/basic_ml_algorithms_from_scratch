"""
- Created By Chris Young in 2019
- This algorithm has the assumption that all features in a dataset is independent and each feature has a Gaussian Distribution
- Bays Theorem: P(target|(x1..xn)) = P((x1...xn)|target) * P(target)/ P((x1...xn))
- P(target|(x1..xn)): priority probability
- P((x1...xn)|target): likelihood given targets
    P((x1...xn)|target) = P(x1|target) * P(x2|target)*....P(xn|target)
- P(target): target prior probability
- P((x1...xn)): features prior probability across the whole dataset
"""
import statistics as st
import scipy.stats as sps
from copy import deepcopy

printdict = lambda x: [print(f'{key}: {x[key]}') for key in x]


class NaiveBays:
    def __init__(self, dataset, dataset_target):
        self.nb_dataset = deepcopy(dataset)
        self.nb_dataset_target = deepcopy(dataset_target)
        self.new_dataset = self.__merge_dataset(self.nb_dataset, self.nb_dataset_target)
        self.data_dict = self.__seperate_data(self.new_dataset)
        self.dataset_length = len(self.nb_dataset)
        self.feature_stats_dic = self.__cal_each_feature_mean_dev()
        self.prior = self.__calc_target_prior_prob()

    # calc prior P(target)
    def __calc_target_prior_prob(self):
        prior = {}
        for key in self.data_dict.keys():
            prior[key] = len(self.data_dict[key]) / self.dataset_length
        return prior

    # calculate mean & std for each feature in the dataset including the targets/labels
    # the result is saved in a dict with label as keys
    def __cal_each_feature_mean_dev(self):
        each_feature_mean_dev_dic = {}
        for each_class in self.data_dict.keys():
            each_feature_mean_dev_dic[each_class] = [(st.mean(feature), st.stdev(feature)) for feature in
                                                     zip(*self.data_dict[each_class])]
        return each_feature_mean_dev_dic

    # put the dataset into a hash table with label as keys
    def __seperate_data(self, dataset):
        seperate = {}
        for each_row in dataset:
            if each_row[-1] not in seperate:
                seperate[each_row[-1]] = []
                seperate[each_row[-1]].append(each_row)
            else:
                seperate[each_row[-1]].append(each_row)
        return seperate

    # merge both data and target into a new dataset for probability calculation
    def __merge_dataset(self, dataset, dataset_target):
        mid_dataset = deepcopy(dataset)
        mid_dataset_target = deepcopy(dataset_target)
        new_dataset = []
        for item in zip(mid_dataset, mid_dataset_target):
            item[0].append(item[1])
            new_dataset.append(item[0])
        return new_dataset

    # P((x1...xn)|target) = P(x1|target) * P(x2|target)*....P(xn|target)
    def __calc_likelihood(self, single_data_set, feature_stats_dic):
        prob_likelihood_evidence = {}
        for single_class in feature_stats_dic.keys():
            prob = 1
            # the target column is not included for this probability calculation
            for index in range(len(feature_stats_dic[single_class]) - 1):
                mean = feature_stats_dic[single_class][index][0]
                devi = feature_stats_dic[single_class][index][1]
                prob *= sps.norm(loc=mean, scale=devi).cdf(single_data_set[index])
            prob_likelihood_evidence[single_class] = prob
        return prob_likelihood_evidence

    def classify(self, single_data, verbose=False):
        single_data = deepcopy(single_data)
        likelihood = self.__calc_likelihood(single_data, self.feature_stats_dic)
        if verbose:
            print('P((x1...xn)|target):')
            printdict(likelihood)
            print('P(target):')
            printdict(self.prior)

        # calc the final probability
        # P((x1...xn))-features prior probability is ignore as it's all same
        res_prob = {}
        for key in likelihood.keys():
            res_prob[key] = likelihood[key] * self.prior[key]
        if verbose:
            print('P(target|(x1..xn)) = P((x1...xn)|target) * P(target):')
            printdict(res_prob)
        res_class = sorted(res_prob, key=lambda x: res_prob[x], reverse=True)[0]
        return res_class

    # datatest evaluate
    def score(self, data_test, data_test_target):
        new_dataset_test = self.__merge_dataset(data_test, data_test_target)
        res_class_lst = []
        for index in range(len(new_dataset_test)):
            res_class_lst.append((self.classify(data_test[index], verbose=False), data_test_target[index]))
        res_class_lst = [1 if pair[1] is pair[0] else 0 for pair in res_class_lst]
        score = 1 - (res_class_lst.count(0) / len(res_class_lst))
        return score

    def classify_ensemble(self, single_data):
        single_data = deepcopy(single_data)
        prob_likelihood_evidence = self.__calc_likelihood(single_data, self.feature_stats_dic)
        # calc the final probability
        res_prob = {}
        for key in prob_likelihood_evidence.keys():
            res_prob[key] = prob_likelihood_evidence[key] * self.prior[key]

        res_class = sorted(res_prob, key=lambda x: res_prob[x], reverse=True)[0]
        return res_class
