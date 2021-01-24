from Decision_Regression_Forest import RegressionForest
import pickle

# load dataset
with open('housing_processed_data','rb') as f:
    dataset = pickle.load(f)
[train_set, train_set_target, test_set, test_set_target] = dataset


# train and save forests model #1
print('model training start....')
# reg_forests = RegressionForest()
# reg_forests.build_forests(train_set, train_set_target, tree_number=50, max_depth=None, max_features=3,
#                           random_patches=True, patches_ratio=0.7, max_estimator=100, parallel=True)
#
# with open('housing_50_100_RegressionForestsModel', 'wb') as f:
#     pickle.dump(reg_forests, f)

# train and save forests model #2
reg_forests_100 = RegressionForest()
reg_forests_100.build_forests(train_set, train_set_target, tree_number=100, max_depth=None, max_features=3,
                          random_patches=True, patches_ratio=0.7, max_estimator=100, parallel=True)

with open('housing_100_100_RegressionForestsModel', 'wb') as f:
    pickle.dump(reg_forests_100, f)


# train and save forests model #3
reg_forests_150 = RegressionForest()
reg_forests_150.build_forests(train_set, train_set_target, tree_number=150, max_depth=None, max_features=3,
                          random_patches=True, patches_ratio=0.7, max_estimator=100, parallel=True)

with open('housing_150_100_RegressionForestsModel', 'wb') as f:
    pickle.dump(reg_forests_150, f)

print('model training done....')