import pickle
import os

pkl_paths = [
    'bus/bus_train_test_names.pkl',
    'busi/busi_train_test_names.pkl',
    'glas/glas_train_test_names.pkl',
    'ham10000/ham10000_train_test_names.pkl',
    'kvasir-instrument/kvasir_train_test_names.pkl'
]

for task_pkl in pkl_paths:
    with open(task_pkl, 'rb') as file:
        loaded_dict = pickle.load(file)

    train_name_list = loaded_dict['train']['name_list']
    test_name_list  = loaded_dict['test']['name_list']

    print(train_name_list)
    print(test_name_list)
    print('train num: {}'.format(len(train_name_list)))
    print('test num: {}'.format(len(test_name_list)))