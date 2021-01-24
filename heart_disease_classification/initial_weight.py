from svm_func import generate_weight
from load_data import data_index

wt_lst = generate_weight(len(data_index)+1, 0, 1)
#print(*wt_lst, sep='\n')
