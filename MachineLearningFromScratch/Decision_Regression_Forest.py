# Developed By Chris Young in 2019
import multiprocessing as mp
from copy import deepcopy
import random
import pandas as pd
import numpy as np
from Decision_Regression_Tree import DecisionTree, RegressionTree


# for classification
class RandomForest:
    def __init__(self):
        self.__forestlist = None
        self.__treefeatures = None

    def set_forest(self, forestlst, treefeatures):
        if isinstance(forestlst, list) and isinstance(treefeatures, list):
            self.__forestlist = forestlst
            self.__treefeatures = treefeatures

        else:
            raise NotImplementedError('Wrong Forest List')

    def get_tree_features(self):
        print(*self.__treefeatures, sep='\n')

    def get_tree_list(self):
        return self.__forestlist

    def print_forest(self):
        print(*self.__forestlist, sep='\n')

    def classify(self, single_dataset, single_dataset_target):
        res_lst = []
        error = 0

        # test trees
        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append((self.__forestlist[tree_index].test(single_test_dataset, single_dataset_target))[0])

        # calc error rate
        res = 1 if res_lst.count(1) >= res_lst.count(0) else 0
        return res, single_dataset_target

    def classify_ensemble(self, single_dataset):
        single_dataset = deepcopy(single_dataset)
        res_lst = []

        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append(self.__forestlist[tree_index].test_notarget(single_test_dataset))

        # calc error rate
        res = 1 if res_lst.count(1) >= res_lst.count(0) else 0
        return res

    def score(self, dataset, dataset_target, parallel=False):

        if not parallel:
            res_lst = []
            for single_data, target in zip(dataset, dataset_target):
                res_lst.append(self.classify(single_data, target))
            error_lst = list(map(lambda x: 1 if x[0] == x[1] else 0, res_lst))
            error_rate = error_lst.count(0) / len(error_lst)
            accuracy = 1 - error_rate
        else:
            pool = mp.Pool(mp.cpu_count())
            res_lst = pool.starmap_async(self.classify, [[single_data, target] for single_data, target in
                                                         zip(dataset, dataset_target)]).get()
            error_lst = list(map(lambda x: 1 if x[0] == x[1] else 0, res_lst))
            error_rate = error_lst.count(0) / len(error_lst)
            accuracy = 1 - error_rate

        return accuracy

    ### random forest build
    # random patches: sampling both training instances and features -> sample columns & rows
    # random subspaces: sampling features of all training instances -> sample columns
    # patches_ratio: to sample ratio percentage of dataset randomly
    # default tree_number is 100
    def build_forests(self, dataset, dataset_target, tree_number=None, max_depth=2, max_features=None,
                      random_subspaces=True,
                      patches_ratio=0.7, parallel=True, multiclass=2):

        dataset = deepcopy(dataset)
        dataset_target = deepcopy(dataset_target)

        if not parallel:
            random.seed = 4
            feature_length = len(dataset[0])
            dataset_height = len(dataset)
            patches_height = int(dataset_height * patches_ratio)
            dataset_frame = pd.DataFrame(dataset)
            forests_lst = []
            tree_features = []

            for i in range(100 if tree_number is None else tree_number):
                patches_dataset = []
                patches_dataset_target = []

                # get the sampling feature index & extract new data
                if max_features is not None and (max_features < 1 or max_features >= feature_length):
                    raise NotImplementedError('improper sample features....')
                else:
                    # extract features and combine all columns to new dataset
                    feature_data_lst = []
                    feature_index = []
                    for j in range(2 if max_features is None else max_features):
                        feature_pos = random.randrange(0, feature_length)
                        feature_index.append(feature_pos)
                        feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
                    new_dataset = np.column_stack(feature_data_lst).tolist()

                    # if random_patches, randomly sample rows of new dataset with certain patches_ratio
                    if not random_subspaces:
                        for k in range(patches_height):
                            patches_pos = random.randrange(0, patches_height)
                            patches_dataset.append(new_dataset[patches_pos])
                            patches_dataset_target.append(dataset_target[patches_pos])

                    # assign new_dataset & target to build trees
                    new_dataset = new_dataset if random_subspaces else patches_dataset
                    new_dataset_target = dataset_target if random_subspaces else patches_dataset_target

                # use new_dataset to build one new tree and save it in forests_lst
                tree_features.append(feature_index)
                decision_tree = DecisionTree()
                decision_tree.build_tree(new_dataset, new_dataset_target, max_depth, multiclass)
                forests_lst.append(decision_tree)
        else:
            forests_lst = []
            tree_features = []
            pool = mp.Pool(mp.cpu_count())
            forests_res = pool.starmap_async(self._build_tree_advanced,
                                             [(dataset, dataset_target, max_depth, max_features, random_subspaces,
                                               patches_ratio, multiclass)
                                              for i in range(100 if tree_number is None else tree_number)]).get()
            pool.close()

            # extract trees & tree_features
            for item in forests_res:
                forests_lst.append(item[0])
                tree_features.append(item[1])

        self.set_forest(forests_lst, tree_features)

    # it's used in the parallel computing
    def _build_tree_advanced(self, dataset, dataset_target, max_depth, max_features, random_subspaces, patches_ratio,
                             multiclass):
        random.seed = 4
        feature_length = len(dataset[0])
        dataset_height = len(dataset)
        patches_height = int(dataset_height * patches_ratio)
        dataset_frame = pd.DataFrame(dataset)

        patches_dataset = []
        patches_dataset_target = []

        # get the sampling feature index & extract new data
        if max_features is not None and (max_features < 1 or max_features >= feature_length):
            raise NotImplementedError('improper sample features....')
        else:
            # extract features and combine all columns to new dataset
            feature_data_lst = []
            feature_index = []
            for j in range(2 if max_features is None else max_features):
                feature_pos = random.randrange(0, feature_length)
                feature_index.append(feature_pos)
                feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
            new_dataset = np.column_stack(feature_data_lst).tolist()

            # if random_patches, randomly sample rows of new dataset with certain patches_ratio
            if not random_subspaces:
                for k in range(patches_height):
                    patches_pos = random.randrange(0, patches_height)
                    patches_dataset.append(new_dataset[patches_pos])
                    patches_dataset_target.append(dataset_target[patches_pos])

            # assign new_dataset & target to build trees
            new_dataset = new_dataset if random_subspaces else patches_dataset
            new_dataset_target = dataset_target if random_subspaces else patches_dataset_target
            decision_tree = DecisionTree()
            decision_tree.build_tree(new_dataset, new_dataset_target, max_depth, multiclass)
            # decision_tree = build_tree(new_dataset, new_dataset_target, max_depth, multiclass)

        return [decision_tree, feature_index]


# for regression
class RegressionForest:
    def __init__(self):
        self.__forestlist = None
        self.__treefeatures = None

    def set_forest(self, forestlst, treefeatures):
        if isinstance(forestlst, list) and isinstance(treefeatures, list):
            self.__forestlist = forestlst
            self.__treefeatures = treefeatures

        else:
            raise NotImplementedError('Wrong Forest List')

    def get_tree_features(self):
        print(*self.__treefeatures, sep='\n')

    def get_tree_list(self):
        return self.__forestlist

    def print_forest(self):
        print(*self.__forestlist, sep='\n')

    def test_target(self, single_dataset, single_dataset_target):
        res_lst = []

        # test trees
        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append(self.__forestlist[tree_index].test(single_test_dataset))

        res = sum(res_lst) / len(res_lst)
        return (res, single_dataset_target)

    def test(self, single_dataset):
        res_lst = []

        # test trees
        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append(self.__forestlist[tree_index].test(single_test_dataset))

        res = sum(res_lst) / len(res_lst)
        return res

    def test_ensemble(self, single_dataset):
        single_dataset = deepcopy(single_dataset)
        res_lst = []

        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append(self.__forestlist[tree_index].test_notarget(single_test_dataset))

        # calc error rate
        res = 1 if res_lst.count(1) >= res_lst.count(0) else 0

        # return (error_rate, result_tupile_list)
        return res

    def error(self, dataset, dataset_target):
        res_lst = []
        for single_data, target in zip(dataset, dataset_target):
            res_lst.append(self.test_target(single_data, target))

        error_lst = list(map(lambda x: abs(x[0] - x[1]), res_lst))
        error = sum(error_lst) / len(error_lst)
        return error

    ### random forest build
    # random patches: sampling both training instances and features -> sample columns & rows
    # random subspaces: sampling features of all training instances -> sample columns
    # patches_ratio: to sample ratio percentage of dataset randomly
    # default tree_number is 100
    def build_forests(self, dataset, dataset_target, tree_number=None, max_depth=2, max_features=None,
                      random_patches=True,
                      patches_ratio=0.7, max_estimator=None, parallel=True, ):
        dataset = deepcopy(dataset)
        dataset_target = deepcopy(dataset_target)

        # parallel computing
        if parallel:
            forests_lst = []
            tree_features = []
            pool = mp.Pool(mp.cpu_count())
            forests_res = pool.starmap_async(self._build_tree_advanced,
                                             [(dataset, dataset_target, max_depth, max_features, random_patches,
                                               patches_ratio, max_estimator)
                                              for i in range(100 if tree_number is None else tree_number)]).get()
            pool.close()

            # extract trees & tree_features
            for item in forests_res:
                forests_lst.append(item[0])
                tree_features.append(item[1])
        else:
            raise NotImplementedError('Regression Forests must have parallel computing...')

        self.set_forest(forests_lst, tree_features)

    # it's used in the parallel computing
    def _build_tree_advanced(self, dataset, dataset_target, max_depth, max_features, random_patches, patches_ratio,
                             max_estimator):
        random.seed = 4
        feature_length = len(dataset[0])
        dataset_height = len(dataset)
        patches_height = int(dataset_height * patches_ratio)
        dataset_frame = pd.DataFrame(dataset)

        patches_dataset = []
        patches_dataset_target = []

        # get the sampling feature index & extract new data
        if max_features is not None and (max_features < 1 or max_features >= feature_length):
            raise NotImplementedError('improper sample features....')
        else:
            # extract features and combine all columns to new dataset
            feature_data_lst = []
            feature_index = []
            for j in range(2 if max_features is None else max_features):
                feature_pos = random.randrange(0, feature_length)
                feature_index.append(feature_pos)
                feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
            new_dataset = np.column_stack(feature_data_lst).tolist()

            # if random_patches, randomly sample rows of new dataset with certain patches_ratio
            # random_subspaces is false, it goes for random_pathces or it keeps random features only
            if random_patches:
                for k in range(patches_height):
                    patches_pos = random.randrange(0, patches_height)
                    patches_dataset.append(new_dataset[patches_pos])
                    patches_dataset_target.append(dataset_target[patches_pos])

            # assign new_dataset & target to build trees
            new_dataset = new_dataset if random_patches else patches_dataset
            new_dataset_target = dataset_target if random_patches else patches_dataset_target
            regression_tree = RegressionTree()
            regression_tree.build_tree(new_dataset, new_dataset_target, max_depth, max_estimator)

        return [regression_tree, feature_index]
