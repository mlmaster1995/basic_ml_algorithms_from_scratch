from copy import deepcopy
import data_process as dp
from Decision_Regression_Forest import RandomForest
from sklearn.ensemble import RandomForestClassifier

# load and generate train & test datasets
data_raw, data_index = dp.load_data('heart_disease_data.csv')
data_rf = data_raw.iloc[:, :-1].values.tolist()
data_rf = dp.normalize_data(data_rf)
data_rf_target = data_raw.iloc[:, -1].values.tolist()
data_train, data_train_target, data_test, data_test_target = \
    dp.split_train_test_data(deepcopy(data_rf), deepcopy(data_rf_target), 0.7, seed=1)

# Random Forests classifier
forests_clf = RandomForest()
forests_clf.build_forests(data_train, data_train_target, tree_number=200, max_features=10, random_subspaces=False,
                          max_depth=None, parallel=True)
print(f'Random Forest acc: {round(forests_clf.score(data_test, data_test_target)*100,3)}%')

# Scikit-Learn Random Forests
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=1, max_features=10)
clf.fit(data_train, data_train_target)
print(f'Scikit-Learn RF acc: {round(clf.score(data_test, data_test_target)*100,5)}%')

