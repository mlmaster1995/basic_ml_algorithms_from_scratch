import pandas as pd
import data_process as dp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from copy import deepcopy
from Linear_regression import LinearRegression


# load data
data_raw, data_index = dp.load_data('housing.csv')

# replace all the text data
data_raw = data_raw.replace({
    'ocean_proximity': {
    '<1H OCEAN': 1,
    'INLAND': 2,
    'NEAR OCEAN': 3,
    'NEAR BAY': 4,
    'ISLAND': 5, }
})

# fill the invalid data
imputer = SimpleImputer(strategy='median')
imputer.fit(data_raw)
X=imputer.transform(data_raw)
data_raw = pd.DataFrame(X, columns=data_index)

# combine data & setup numeric categary of data
data_raw['rooms_per_houshold'] = data_raw['total_rooms']/data_raw['households']
data_raw['bedrooms_per_room'] = data_raw['total_bedrooms']/data_raw['total_rooms']
data_raw['population_per_household'] = data_raw['population']/data_raw['households']

# clearning data
data_raw = data_raw.drop(['households', 'total_bedrooms',
                          'ocean_proximity', 'population_per_household',
                          'population', 'longitude'], axis=1)

# sperate training data and target data
data_target = [deepcopy(data_raw.iloc[:, 4].tolist())]
data_raw = data_raw.drop(columns=['median_house_value']).values.tolist()

# scale dataset
data_target_scale = (dp.normalize_data(data_target))[0]
data_raw_scale = dp.normalize_data(data_raw)

# seperate train, test dataset
train_set, test_set, train_set_target, test_set_target = \
    train_test_split(data_raw_scale, data_target_scale, test_size=0.2, random_state=1)

# linear regression
linear_regression = LinearRegression(train_set, train_set_target)
linear_regression.train_model(reg=1e-5)
error = linear_regression.calc_error(test_set, test_set_target)
print(f'lr error:{error}')

# sklearn linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(train_set, train_set_target)

error=0
for val, target in list(zip(test_set,test_set_target)):
    error += abs(target-reg.predict([val]))
error /= len(test_set)
print(f'sklearn lr error:{error}')

# sklearn svm regression
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=0.001)
svm_reg.fit(train_set, train_set_target)

error_lst = list(map(lambda pair: abs(svm_reg.predict([pair[0]])-pair[1]),
                     list(zip(test_set, test_set_target))))
error = sum(error_lst)/len(error_lst)
print(f'sklearn svm error:{error}')

# sklearn svm kernel regression
from sklearn.svm import SVR
svm_ker = SVR(kernel='rbf', C=100, epsilon=0.001, gamma='auto')
svm_ker.fit(train_set, train_set_target)

error_lst = list(map(lambda pair: abs(svm_ker.predict([pair[0]])-pair[1]),
                     list(zip(test_set, test_set_target))))
error = sum(error_lst)/len(error_lst)
print(f'sklearn svm kernel error:{error}')

# sklearn gradient boost regression
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=100, n_estimators=200, learning_rate=0.1, min_samples_split=150)
gbrt.fit(train_set, train_set_target)

error = 0
for val, tar in list(zip(test_set, test_set_target)):
    res = gbrt.predict([val])
    error += abs(res-tar)
print(f'sklearn gradient boost error:{error/len(test_set)}')

# sklearn regression tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(min_samples_split=150)
tree.fit(train_set, train_set_target)

error = 0
for val, tar in list(zip(test_set, test_set_target)):
    res = tree.predict([val])
    error += abs(res-tar)
print(f'sklearn regression tree error:{error/len(test_set)}')


print('done...')































