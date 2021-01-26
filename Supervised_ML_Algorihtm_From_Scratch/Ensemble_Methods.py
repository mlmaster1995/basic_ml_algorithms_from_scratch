
# Created By Chris Young in 2019
# Harding Volting and Adaboost are developed
# soft_voting, stacking, bagging, gradient boost not finished yet
from copy import deepcopy
from Decision_Regression_Forest import RandomForest
from KNN import KNN
from Logistic_Regression import Logistic_Regression
from Naive_Bays import NaiveBays
from SVM import SVM_Kernel, SVM_Linear
from Decision_Regression_Tree import DecisionTree
import numpy as np
import math
import random
from random import random


class EnsembleLearner:
    def __init__(self, dataset, dataset_target):
        self.dataset = deepcopy(dataset)
        self.dataset_target = deepcopy(dataset_target)
        # hard volting parameters
        self.__algorithm_lst = None
        # adaboost parameters
        self.number_estimator = None
        self.boost_model = None
        self.predict_list = None
        self.estimator_limit = 2000

    # ??
    def soft_voting(self):
        pass

    # ??
    def stacking(self):
        pass

    # similar as the random forest
    def bagging(self):
        pass

    # ??
    def gradientBoost(self):
        pass

    # hard volting predict func
    def hard_voting_predict(self, single_dataset, *argv):
        res_lst = []
        single_dataset = deepcopy(single_dataset)
        self.__algorithm_lst = argv
        if self.__classtype_check():
            for alg in argv:
                res_lst.append(alg.classify_ensemble(single_dataset))
        else:
            raise NotImplementedError('Unknown algorithm...')
        print(res_lst)
        res = 1 if res_lst.count(1)>=res_lst.count(0) else 0
        return res

    # hard volting score func
    def hard_voting_score(self, datatest, datatest_target, *argv):
        self.__algorithm_lst = argv
        datatest = deepcopy(datatest)
        datatest_target = deepcopy(datatest_target)
        res_tupile=[]
        if self.__classtype_check():
            for index in range(len(datatest)):
                res_lst = []
                for alg in argv:
                    res_lst.append(alg.classify_ensemble(datatest[index]))
                # print(res_lst)
                res = 1 if res_lst.count(1) >= res_lst.count(0) else 0
                res_tupile.append((res, datatest_target[index]))
        else:
            raise NotImplementedError('Unknown algorithm...')
        res_final = list(map(lambda x: 1 if x[0]==x[1] else 0, res_tupile))
        error = res_final.count(0)/len(res_tupile)
        score = 1-error
        return score

    # hard volting algorithm type check
    def __classtype_check(self):
        for alg_type in self.__algorithm_lst:
            if isinstance(alg_type, RandomForest):
                continue
            elif isinstance(alg_type, KNN):
                continue
            elif isinstance(alg_type, Logistic_Regression):
                continue
            elif isinstance(alg_type, NaiveBays):
                continue
            elif isinstance(alg_type, SVM_Kernel):
                continue
            elif isinstance(alg_type, SVM_Linear):
                continue
            elif isinstance(alg_type, DecisionTree):
                continue
            else:
                return False
        return True

    # adaBoost training func
    def adaBoost_train(self, estimator_limit=None, learning_rate=1, base_estimator='decision tree'):
        # basic model parameters
        boost_model = []
        predictor_weight_list = []
        estimator_count = 0
        # traning model
        estimator_limit = self.estimator_limit if estimator_limit is None else estimator_limit
        while estimator_count < estimator_limit:
            # initial estimator parameters
            estimator_count += 1
            # fill up the model training & testing data
            if estimator_count == 1:
                estimator_data = self.dataset
                estimator_target = self.dataset_target
            else:
                if len(misclf_index_list) == 0:
                    break
                else:
                    estimator_data, estimator_target = self.__weighted_sampling(self.dataset,
                                                                                self.dataset_target,
                                                                                instance_weight_rate_list,
                                                                                len(self.dataset))

            # train the model with base estimator and test it
            # terror_lst is to recorder which sample is correct
            # misclf_index_list is to recorder the misclassified sample index
            terror_lst = [1] * len(self.dataset)
            misclf_index_list = list(range(0, len(self.dataset)))
            if base_estimator == 'decision tree':
                tree = DecisionTree()
                tree.build_tree(estimator_data, estimator_target, max_depth=1, multiclass=2)
                for index in range(len(self.dataset)):
                    res = tree.test(self.dataset[index], self.dataset_target[index])
                    if res[0] == res[1]:
                        terror_lst[index] = 0
                        misclf_index_list.remove(index)
                    else:
                        terror_lst[index] = 1

            # initiate weights and calculate the error
            instance_weight_rate = 1 / len(self.dataset)
            instance_weight_rate_list = [instance_weight_rate] * len(self.dataset)
            error = sum(np.multiply(instance_weight_rate_list, terror_lst).tolist()) / sum(instance_weight_rate_list)

            # calc predictor weight
            if 1 > error > 0:
                predictor_weight_list.append(learning_rate * math.log((1 - error) / error))
            elif error == 0:
                predictor_weight_list.append(max(predictor_weight_list)+0.5 if len(predictor_weight_list) > 0 else 10)
                boost_model.append((tree, predictor_weight_list[-1]))
                break
            else:
                raise NotImplementedError('Zero classification and check the algorithm...')

            # update instance weight
            temp_array = np.array(deepcopy(instance_weight_rate_list))
            instance_weight_rate_list = \
                (np.array(instance_weight_rate_list) +
                 np.multiply(temp_array * math.exp(predictor_weight_list[-1]), np.array(terror_lst))).tolist()

            # normalize all instance weight
            sum_instance_weight = deepcopy(sum(instance_weight_rate_list))
            instance_weight_rate_list = (np.array(instance_weight_rate_list)/sum_instance_weight).tolist()

            # save the tree
            boost_model.append((tree, predictor_weight_list[-1]))

        # update the estimator number
        if estimator_count == self.estimator_limit:
            print('Maximum estimator limit is reached...')
        self.number_estimator = estimator_count
        self.boost_model = boost_model

    # adaBoost predicting func
    def adaBoost_predict(self, single_dataset):
        predict_sum = 0
        predict_lst=[]
        if self.boost_model is not None:
            for model in self.boost_model:
                scale = model[-1]
                base_model = model[0]
                res = base_model.classify_ensemble(single_dataset)
                res = -1 if res == 0 else 1
                predict_lst.append((res, scale))
                predict_sum += res*scale

            predict_lst.append(predict_sum)
            self.predict_list = predict_lst
            predict_res = 1 if predict_sum>0 else 0
            return predict_res
        else:
            raise NotImplementedError('AdaBoost model is empty...')

    # adaBoost score func
    def adaBoost_score(self, datatest, datatest_target):
        error = 0
        for index in range(len(datatest)):
            res = self.adaBoost_predict(datatest[index])

            if res != datatest_target[index]:
                error += 1

            # print((res, datatest_target[index]))
        error_rate = error/len(datatest)
        accuracy = 1-error_rate
        return accuracy

    # adaBoost weighted resampling func
    def __weighted_sampling(self, data, data_target, sample_weight, resample_size):
        sample_index_lst = []
        new_sample = []
        new_sample_target = []

        weights = deepcopy(sample_weight)
        CDF_weights = (np.array(weights).cumsum()).tolist()
        for _ in range(resample_size):
            ref = random()
            for i in range(len(sample_weight)):
                if ref < CDF_weights[i]:
                    sample_index_lst.append(deepcopy(i))
                    break

        for index in sample_index_lst:
            new_sample.append(deepcopy(data[index]))
            new_sample_target.append(deepcopy(data_target[index]))

        return new_sample, new_sample_target








