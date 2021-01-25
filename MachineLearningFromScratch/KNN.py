"""Created by Chris Young in 2019"""
import math
import time
import multiprocessing as mp


class KNN:
    def __init__(self, dataset, dataset_target):
        self.knn_dataset = dataset
        self.knn_dataset_target = dataset_target
        self.__distance_list = []
        self.__standardize_distance = []

    def __clear_lst(self):
        self.__distance_list = []
        self.__standardize_distance = []

    # distance calculation recursion
    def __calc_distance_recursion(self, single_dataset, index=0):
        ele_sqrt = [math.pow((pair[0] - pair[1]), 2) for pair in zip(single_dataset, self.knn_dataset[index])]
        distance = math.sqrt(sum(ele_sqrt))
        self.__distance_list.append(distance)

        if index + 1 is not len(self.knn_dataset):
            index += 1
            self.__calc_distance_recursion(single_dataset, index)
        else:
            return None

    # normalize distance
    def __normalize_distance(self):
        if len(self.__distance_list) == 0:
            raise NotImplementedError('No distance is calculated...')
        else:
            distance_min = min(self.__distance_list)
            distance_max = max(self.__distance_list)
            self.__standardize_distance = \
                [(item - distance_min) / (distance_max - distance_min) for item in self.__distance_list]

    # classify single dataset
    def classify(self, single_dataset, K, recursion=False, showtime=False):
        startime = time.time()
        # clear out all the distance list
        self.__clear_lst()

        if not recursion:
            # fill out the distance list with respect to the single_dataset
            # standardize the distance
            self.__calc_distance_recursion(single_dataset, index=0)
            self.__normalize_distance()

            # make tuple list of (distance, target_class)
            distance_tuple_lst = [(self.__standardize_distance[index], self.knn_dataset_target[index]) for index in
                                  range(len(self.knn_dataset_target))]
            sorted_distance_tuple_lst = sorted(distance_tuple_lst, key=lambda x: x[0])

            # count class number
            if (len(self.knn_dataset) - 1) >= K > 0:
                target_count = {}
                for tuple_pair in sorted_distance_tuple_lst[:K]:
                    if tuple_pair[1] not in target_count:
                        target_count[tuple_pair[1]] = 1
                    else:
                        target_count[tuple_pair[1]] += 1
            else:
                raise NotImplementedError('Improper K value')

            # output the class
            class_key = sorted(target_count, key=lambda x: target_count[x], reverse=True)[0]

        else:
            # calculate the distance between input data and sample data
            for single_data in self.knn_dataset:
                ele_sqrt = [math.pow((pair[0] - pair[1]), 2) for pair in zip(single_dataset, single_data)]
                distance = math.sqrt(sum(ele_sqrt))
                self.__distance_list.append(distance)
            # normalize the distance
            self.__normalize_distance()

            # make tuple list of (distance, target_class)
            distance_tuple_lst = [(self.__standardize_distance[index], self.knn_dataset_target[index]) for index in
                                  range(len(self.knn_dataset_target))]
            sorted_distance_tuple_lst = sorted(distance_tuple_lst, key=lambda x: x[0])

            # count class number
            if (len(self.knn_dataset) - 1) >= K > 0:
                target_count = {}
                for tuple_pair in sorted_distance_tuple_lst[:K]:
                    if tuple_pair[1] not in target_count:
                        target_count[tuple_pair[1]] = 1
                    else:
                        target_count[tuple_pair[1]] += 1
            else:
                raise NotImplementedError('Improper K value')

            # output the class
            class_key = sorted(target_count, key=lambda x: target_count[x], reverse=True)[0]

        if showtime:
            print(f'run time:{round(time.time() - startime, 5)} sec')

        return class_key

    # evaulate a dataset and parallel computing is implemented
    def evaulate(self, data_test, data_test_target, K, recursion=False, parallel=True):
        if parallel:
            pool = mp.Pool(mp.cpu_count())
            class_key_list = pool.starmap_async(self.classify, [(data, K, recursion) for data in data_test]).get()
            pool.close()

        else:
            class_key_list = []
            for data in data_test:
                class_key_list.append(self.classify(data, K, recursion))

        error_list = [1 if pair[0] == pair[1] else 0 for pair in zip(class_key_list, data_test_target)]
        error = error_list.count(0) / len(error_list)

        return (1 - error), K

    # search the best K
    def optimize_K(self, data_test, data_test_target, showtime=False):
        startime = time.time()
        max_K = len(self.knn_dataset) - 1
        pool = mp.Pool(mp.cpu_count())
        score_list = pool.starmap_async(self.evaulate, [(data_test, data_test_target, i, False, False) for i in
                                                        list(range(1, max_K))]).get()
        pool.close()

        optimum_K = sorted(score_list, key=lambda x: x[0], reverse=True)[0]
        if showtime:
            print(f'process time: {round(time.time() - startime, 5)} sec')

        return optimum_K
