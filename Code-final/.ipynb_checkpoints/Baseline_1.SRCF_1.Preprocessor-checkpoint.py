import os
import time
import wandb
import random
import numpy as np
import utils.io as uio
import core.utils as cuil
import matplotlib.pyplot as plt

from tqdm import tqdm
from configs.srcf_configs import Config
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def execute(config, input_data):

    start = time.time()
    
    features = generate_features(input_data, input_data)

    end = time.time()
            
    run_time = (end - start)
    
    print("avg_time: ",run_time/input_data.shape[0])

    return features


def generate_features(target, base):
    input_data = []
    for sample in tqdm(target):

        input_data.append(distance_features(sample, base))

    input_data = np.array(input_data)

    return input_data


def distance_features(sample, base):
    input_features = []

    for series in base:

        input_features.append(np.sqrt(np.sum((sample - series)**2)))

    input_features = np.array(input_features)

    return input_features


def split(config, training_data, testing_data, training_label, testing_label, features_index):
    
    output_data = {
        "training": training_data[:,features_index],
        "testing": testing_data[:,features_index],
        "training_label": training_label,
        "testing_label": testing_label
    }

    return output_data


if __name__ == "__main__":

    for ratio in range(1, 11):
        
        series_ratio = ratio/10
        
        config = Config(message="20210120", total_length=5000, dataset_name="ptbxl", model_name="PTBXL-O")

        print("Here ratio: ", ratio, " dataset: ", config.dataset_name, "length: ",config.total_length, "series_ratio: ",series_ratio)
        
        input_path = os.path.join(config.root_dir,
                                  config.data_dir,
                                  config.dataset_name) + "/"
        tmp_path = os.path.join(config.root_dir,
                                config.data_dir,
                                config.dataset_name,
                                config.tmp_dir)
        '''
        data, raw_labels, labels, classes = uio.load_formmated_raw_data(
            input_path, "all", tmp_path)
        '''
        
        data, raw_labels, labels, classes = uio.load_formmated_raw_data(
            input_path, "superdiagnostic", tmp_path)

        raw_length = uio.get_length(data)
        input_data = uio.resize(data, config.total_length, series_ratio)
        labels = labels
        raw_labels = raw_labels
        #input_data = uio.norm(input_data)
        y = np.argmax(raw_labels, axis=1)
        
        
        
        feature_data = execute(config, input_data)
        #feature_data = input_data
        for seed in range(1, 11):

            print('seed:',seed)
            state_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.state_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed),
                                      str(ratio))
            uio.check_folder(state_path)
                        
            if config.dataset_name == "ptbxl":
                training_data = feature_data[labels.strat_fold!=seed]
                testing_data = feature_data[labels.strat_fold==seed]
                
                training_label = raw_labels[labels.strat_fold!=seed]
                testing_label = raw_labels[labels.strat_fold==seed]
                
                training_length = raw_length[labels.strat_fold!=seed]
                testing_length = raw_length[labels.strat_fold==seed]
                
                print(labels.index[labels.strat_fold!=seed].shape)
                
                print(training_data.shape)
                
                training_features_index =  (labels.strat_fold!=seed).tolist()
                                
            else:
            
                sss = StratifiedKFold(n_splits=10, random_state=0)

                sss.get_n_splits(feature_data, y)

                training_features_index = None

                for index, (train_index, test_index) in enumerate(sss.split(feature_data, y)):
                    if(index is (seed-1)):
                        print("Runing:",seed)
                        break
        
                
                training_data, testing_data = feature_data[train_index], feature_data[test_index]
                training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
                training_length, testing_length = raw_length[train_index], raw_length[test_index]
                training_features_index = train_index

            output_data = split(config,
                                training_data,
                                testing_data,
                                training_label,
                                testing_label,
                                training_features_index)

            uio.save_pkfile(state_path+"/"+config.state_name, output_data)
        

state_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.state_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(1),
                                      str(ratio))
a = uio.load_pkfile(state_path+"/"+config.state_name)

# if __name__ == "__main__":
#     
#     for ratio in range(1, 2):
#         
#         series_ratio = ratio/10/2
#         
#         config = Config(message="20210120", total_length=30000, dataset_name="ICBEB")
#
#         input_path = os.path.join(config.root_dir,
#                                   config.data_dir,
#                                   config.dataset_name) + "/"
#         tmp_path = os.path.join(config.root_dir,
#                                 config.data_dir,
#                                 config.dataset_name,
#                                 config.tmp_dir)
#
#         data, raw_labels, labels, classes = uio.load_formmated_raw_data(
#             input_path, "all", tmp_path)
#
#         raw_length = uio.get_length(data)
#
#         input_data = uio.resize(data, config.total_length, series_ratio)
#
#         print(input_data.shape)
#
#         input_data = uio.norm(input_data)
#         y = np.argmax(raw_labels, axis=1)
#
#         sss = StratifiedShuffleSplit(
#                 n_splits=1, test_size=0.1, random_state=0)
#
#         sss.get_n_splits(input_data, y)
#
#         for train_index, test_index in sss.split(input_data, y):
#             training_data, testing_data = input_data[train_index], input_data[test_index]
#             training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
#             training_length, testing_length = raw_length[train_index], raw_length[test_index]
#                     
#         generate_features(testing_data, training_data)


