import pickle

# load dataset
with open('housing_processed_data','rb') as f:
    dataset = pickle.load(f)
[train_set, train_set_target, test_set, test_set_target] = dataset

# load model
names =['housing_50_100_RegressionForestsModel','housing_100_100_RegressionForestsModel','housing_150_100_RegressionForestsModel']
for nm in names:
    with open(nm, 'rb') as f:
        reg_forests = pickle.load(f)

    # test model
    error = reg_forests.error(test_set, test_set_target)
    print(f'{nm} error & accuracy')
    print(f'error: {round(error,2)}')
    print(f'accuracy:{round((1-error),2)*100}%\n')