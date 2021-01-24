import math
import statistics as st

printdict = lambda x: [print(f'{key}: {x[key]}') for key in x]

## gaussian function
def Gaussian_calc(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


## split the model data into different classes within one map (dictionary)
def seperate_data(data_model):
    seperate={}
    for each_row in data_model:
        if each_row[-1] not in seperate:
            seperate[each_row[-1]]=[]
            seperate[each_row[-1]].append(each_row)
        else:
            seperate[each_row[-1]].append(each_row)
    return seperate


## calculate mean & std for each evidence in each class in the data_model
def feature_mean_devi(data_set):
    data_dict = seperate_data(data_set)
    feature_stats_dic={}
    for each_class in data_dict.keys():
        feature_stats_dic[each_class]=[(st.mean(feature), st.stdev(feature)) for feature in zip(*data_dict[each_class])]
    return feature_stats_dic


## calc probability of likelihood of evidence
## P(X|Y)=P(x1|Y)*P(x2|Y)*P(x3|Y)*P(x1|Y)... based on the class for single-data
def calc_prob_likelihood_evidence(single_data_set, feature_stats_dic):
    prob_likelihood_evidence = {}

    for single_class in feature_stats_dic.keys():
        prob = 1

        # no target column is for the calculation
        for index in range(len(feature_stats_dic[single_class]) - 1):
            mean = feature_stats_dic[single_class][index][0]
            devi = feature_stats_dic[single_class][index][1]
            prob *= Gaussian_calc(single_data_set[index], mean, devi)

        prob_likelihood_evidence[single_class] = prob
    return prob_likelihood_evidence


## calc prior P(Y)
def calc_prior(dataset):
    data_dict = seperate_data(dataset)
    data_all_length = len(dataset)
    prior = {}
    for key in data_dict.keys():
        prior[key] = len(data_dict[key]) / data_all_length

    return prior




