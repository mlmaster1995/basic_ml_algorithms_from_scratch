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
print(f'error:{error}')
print(f'reg score: {1-error}')

# sklearn linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(train_set, train_set_target)
score = reg.score(test_set, test_set_target)
print(f'sklearn lr score:{score}')

# sklearn svm regression
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=0.001)
svm_reg.fit(train_set, train_set_target)
score = svm_reg.score(train_set, train_set_target)
print(f'svr score:{score}')

































