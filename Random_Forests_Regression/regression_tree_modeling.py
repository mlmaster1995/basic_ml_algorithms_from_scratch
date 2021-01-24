from Decision_Regression_tree import RegressionTree
import pickle

# load dataset
with open('housing_processed_data', 'rb') as f:
    dataset = pickle.load(f)
[train_set, train_set_target, test_set, test_set_target] = dataset

# train and save the model
print('model training start....')
reg_tree = RegressionTree()
reg_tree.build_tree(train_set, train_set_target, max_depth=None, max_estimator=150)

with open('housing_150_model_RegressionTree', 'wb') as f:
    pickle.dump(reg_tree, f)

print('model training done....')













