import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# load the data
with open('housing_processed_data','rb') as f:
    [train_set, train_set_target, test_set, test_set_target] = pickle.load(f)

# training gradient decent model
gbrt = GradientBoostingRegressor(max_depth=100, n_estimators=200, learning_rate=0.1, min_samples_split=150)
gbrt.fit(train_set, train_set_target)

filename = 'housing_gradient_boost_model'
pickle.dump(gbrt, open(filename, 'wb'))

# testing
model = pickle.load(open(filename, 'rb'))

error = 0
for val, tar in list(zip(train_set, train_set_target)):
    res = model.predict([val])
    error += abs(res-tar)
print(f'train error:{error/len(train_set)}')

error = 0
for val, tar in list(zip(test_set, test_set_target)):
    res = model.predict([val])
    error += abs(res-tar)
print(f'test error:{error/len(test_set)}')

# print('******************************************')
#
# # training decision tree
# tree = DecisionTreeRegressor(min_samples_split=150)
# tree.fit(train_set, train_set_target)
#
# error = 0
# for val, tar in list(zip(train_set, train_set_target)):
#     res = tree.predict([val])
#     error += abs(res-tar)
# print(f'train error:{error/len(train_set)}')
#
# error = 0
# for val, tar in list(zip(test_set, test_set_target)):
#     res = tree.predict([val])
#     error += abs(res-tar)
# print(f'test error:{error/len(test_set)}')





