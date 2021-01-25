from data_process import data_put_all


split_ratio = 0.8
filename = 'heart_disease_data.csv'
dt_train, dt_train_target, dt_test, dt_test_target, data_index = \
    data_put_all(filename, split_ratio, rand=False)


#print(*dt_train, sep='\n')