import pickle

# load dataset
with open('housing_processed_data','rb') as f:
    dataset = pickle.load(f)
[train_set, train_set_target, test_set, test_set_target] = dataset

# load model
with open('housing_trained_model_RegressionTree', 'rb') as f:
    reg_tree_none = pickle.load(f)
# test model
print(f'no restricted model test:')
error = reg_tree_none.error(train_set, train_set_target)
print(f'train set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')
error = reg_tree_none.error(test_set, test_set_target)
print(f'test set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')

# load model
with open('housing_100_model_RegressionTree', 'rb') as f:
    reg_tree_100 = pickle.load(f)
# test model
print(f'\n100 max_estimator model test:')
error = reg_tree_100.error(train_set, train_set_target)
print(f'train set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')
error = reg_tree_100.error(test_set, test_set_target)
print(f'test set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')

# load model
with open('housing_150_model_RegressionTree', 'rb') as f:
    reg_tree_150 = pickle.load(f)
# test model
print(f'\n150 max_estimator model test:')
error = reg_tree_150.error(train_set, train_set_target)
print(f'train set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')
error = reg_tree_150.error(test_set, test_set_target)
print(f'test set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')


# load model
with open('housing_200_model_RegressionTree', 'rb') as f:
    reg_tree_200 = pickle.load(f)
# test model
print(f'\n200 max_estimator model test:')
error = reg_tree_200.error(train_set, train_set_target)
print(f'train set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')
error = reg_tree_200.error(test_set, test_set_target)
print(f'test set error & accuracy: ')
print(f'error:{round(error,2)}')
print(f'accuracy:{round((1-error)*100,2)}%')