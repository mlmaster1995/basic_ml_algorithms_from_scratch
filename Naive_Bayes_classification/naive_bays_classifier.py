import statistics as st
import math

printdict = lambda x: [print(f'{key}: {x[key]}') for key in x]

class NaiveBays:
    def __init__(self, dataset, dataset_target):
        self.nb_dataset = dataset
        self.nb_dataset_target = dataset_target
        self.new_dataset = self.__merge_dataset(self.nb_dataset, self.nb_dataset_target)
        self.data_dict = self.__seperate_data(dataset.copy())
        self.dataset_length = len(dataset)
        self.feature_stats_dic = self.__feature_mean_devi()
        self.prior = self.__calc_prior()

    def classify(self, single_data, prt=False):
        # new_dataset = self.__merge_dataset(self.nb_dataset, self.nb_dataset_target)
        # feature_stats_dic = self.__feature_mean_devi()
        prob_likelihood_evidence = self.__calc_prob_likelihood_evidence(single_data, self.feature_stats_dic)
        if prt:
            print('P(X|Y):')
            printdict(prob_likelihood_evidence)

        # prior = self.__calc_prior()
        if prt:
            print('P(Y):')
            printdict(self.prior)

        # calc the final probability
        res_prob = {}
        for key in prob_likelihood_evidence.keys():
            res_prob[key] = prob_likelihood_evidence[key] * self.prior[key]

        if prt:
            print('P(Y|X)=P(X|Y)*P(Y):')
            printdict(res_prob)

        res_class= sorted(res_prob, key=lambda x: res_prob[x], reverse=True)[0]
        return res_class

    # datatest score
    def score(self, data_test, data_test_target):
        new_dataset_test = self.__merge_dataset(data_test, data_test_target)
        res_class_lst=[]
        for index in range(len(new_dataset_test)):
            res_class_lst.append((self.classify(data_test[index], prt=False), data_test_target[index]))

        res_class_lst = [1 if pair[1] is pair[0] else 0 for pair in res_class_lst]
        score = 1-(res_class_lst.count(0)/len(res_class_lst))

        return score


    ## calc prior P(Y)
    def __calc_prior(self):
        # data_dict = self.__seperate_data(dataset)
        data_all_length = self.dataset_length
        prior = {}
        for key in self.data_dict.keys():
            prior[key] = len(self.data_dict[key]) / data_all_length

        return prior

    def __merge_dataset(self, dataset, dataset_target):
        new_dataset = []
        for item in zip(dataset, dataset_target):
            item[0].append(item[1])
            new_dataset.append(item[0])
        return new_dataset

    ## calculate mean & std for each evidence in each class in the data_model
    def __feature_mean_devi(self):
        # data_dict = self.__seperate_data(data_set)
        feature_stats_dic = {}
        for each_class in self.data_dict.keys():
            feature_stats_dic[each_class] = [(st.mean(feature), st.stdev(feature)) for feature in
                                             zip(*self.data_dict[each_class])]
        return feature_stats_dic


    ## split the model data into different classes within one map (dictionary)
    def __seperate_data(self, data_model):
        seperate = {}
        for each_row in data_model:
            if each_row[-1] not in seperate:
                seperate[each_row[-1]] = []
                seperate[each_row[-1]].append(each_row)
            else:
                seperate[each_row[-1]].append(each_row)
        return seperate


    ## calc probability of likelihood of evidence
    ## P(X|Y)=P(x1|Y)*P(x2|Y)*P(x3|Y)*P(x1|Y)... based on the class for single-data
    def __calc_prob_likelihood_evidence(self, single_data_set, feature_stats_dic):
        prob_likelihood_evidence = {}

        for single_class in feature_stats_dic.keys():
            prob = 1

            # no target column is for the calculation
            for index in range(len(feature_stats_dic[single_class]) - 1):
                mean = feature_stats_dic[single_class][index][0]
                devi = feature_stats_dic[single_class][index][1]
                prob *= self.__Gaussian_calc(single_data_set[index], mean, devi)

            prob_likelihood_evidence[single_class] = prob
        return prob_likelihood_evidence


    ## gaussian function
    def __Gaussian_calc(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        res = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
        return res















