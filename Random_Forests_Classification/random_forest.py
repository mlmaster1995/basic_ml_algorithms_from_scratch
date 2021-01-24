import multiprocessing as mp

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
        res_lst= []
        error= 0

        # test trees
        for tree_index in range(len(self.__forestlist)):
            sample_features = self.__treefeatures[tree_index]
            single_test_dataset = list(map(lambda x: single_dataset[x], sample_features))
            res_lst.append((self.__forestlist[tree_index].test(single_test_dataset, single_dataset_target))[0])

        # calc error rate
        res = 1 if res_lst.count(1)>=res_lst.count(0) else 0

        # return (error_rate, result_tupile_list)
        return (res, single_dataset_target)

    def score(self, dataset, dataset_target, parallel=False):

        if not parallel:
            res_lst=[]
            for single_data, target in zip(dataset, dataset_target): res_lst.append(self.classify(single_data, target))
            error_lst = list(map(lambda x: 1 if x[0]==x[1] else 0,res_lst))
            error_rate = error_lst.count(1)/len(error_lst)
            accuracy = 1-error_rate
        else:
            pool = mp.Pool(mp.cpu_count())
            res_lst = pool.starmap_async(self.classify, [[single_data, target] for single_data, target in zip(dataset, dataset_target)]).get()
            error_lst = list(map(lambda x: 1 if x[0] == x[1] else 0, res_lst))
            error_rate = error_lst.count(1) / len(error_lst)
            accuracy = 1 - error_rate

        return accuracy