from Random_forests_func import build_tree
from data_process import load_data, split_train_test_data
from Random_forests_func import build_forests
import time
import multiprocessing as mp

startime=time.time()

# from data_import import dataset, dataset_target
## load data

data_raw = load_data('heart_disease_data.csv')
data_raw = data_raw[0]
data_raw_index = list(data_raw.columns)[0:-1]
data_rf = data_raw.iloc[:, 0:-1].values.tolist()
data_rf_target = data_raw.iloc[:, -1].values.tolist()
data_train, data_train_target, data_test, data_test_target= split_train_test_data(data_rf, data_rf_target, 0.7, seed=2)

## decision_tree build
# decision_tree = build_tree(data_train, data_train_target, max_depth=None)
# decision_tree.print_nodes()
# score = decision_tree.score(data_test, data_test_target)
# print(score)

# 100 times loop for testing

'''
loop=100
score_lst=[]
for i in range(loop):
    startime = time.time()
    print(f'#{i} loop processing...')

    data_raw = load_data('heart_disease_data.csv')
    data_raw = data_raw[0]
    data_raw_index = list(data_raw.columns)[0:-1]
    data_rf = data_raw.iloc[:, 0:-1].values.tolist()
    data_rf_target = data_raw.iloc[:, -1].values.tolist()
    data_train, data_train_target, data_test, data_test_target = split_train_test_data(data_rf, data_rf_target, 0.7, rand=True)
    forests = build_forests(data_train, data_train_target, tree_number=500, max_features=10, random_subspaces=False, max_depth=None, parallel=True)

    print(f'Done, {round(time.time() - startime, 2)} seconds! Score: {round(forests.score(data_test, data_test_target)*100, 2)}%')
    score_lst+= [forests.score(data_test, data_test_target)]
print('min:', min(score_lst), 'max:', max(score_lst))
'''

'''
def test(i):
    startime = time.time()
    print(f'#{i} processing start...')
    data_raw = load_data('heart_disease_data.csv')
    data_raw = data_raw[0]
    data_rf = data_raw.iloc[:, 0:-1].values.tolist()
    data_rf_target = data_raw.iloc[:, -1].values.tolist()
    data_train, data_train_target, data_test, data_test_target = split_train_test_data(data_rf, data_rf_target, 0.7, rand=True)
    forests = build_forests(data_train, data_train_target, tree_number=500, max_features=10, random_subspaces=False, max_depth=None, parallel=False)
    print(f'#{i} Done, {round(time.time()-startime,2)}seconds! Score: {forests.score(data_test, data_test_target)}')
    return forests.score(data_test, data_test_target)

pool = mp.Pool(mp.cpu_count())
score_lst = pool.map_async(test, [i for i in range(100)]).get()
pool.close()
print('min:', min(score_lst), 'max:', max(score_lst))
'''

## random forests
forests = build_forests(data_train, data_train_target, tree_number=500, max_features=10, random_subspaces=False, max_depth=None)
# forests.get_tree_list()[0].print_nodes()
# res= forests.classify([67.0, 0.0, 0.0, 106.0, 223.0, 0.0, 1.0, 142.0, 0.0, 0.3, 2.0, 2.0, 2.0], 1)
# print(res)
print(forests.score(data_test, data_test_target))
print(f'processing time: {round(time.time()-startime,2)} seconds')


## random forests parallel computing
# forests = build_forests(data_train, data_train_target, tree_number=500, max_features=10, random_subspaces=False, max_depth=None, parallel=True)
# print(forests.score(data_test, data_test_target, parallel=False))
# print(f'processing time: {round(time.time()-startime,2)} seconds')








