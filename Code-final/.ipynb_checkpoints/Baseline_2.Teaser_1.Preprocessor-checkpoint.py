import os
import time
import wandb
import random
import numpy as np
import utils.io as uio
import core.utils as cuil
import matplotlib.pyplot as plt

from tqdm import tqdm
from configs.teaser_config import Config
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from pyts.multivariate.transformation import WEASELMUSE
from sklearn.preprocessing import StandardScaler

def execute(config, training_data, training_label, testing_data, testing_label, is_multiple_label=False):

    cut_point = int(config.total_length * config.series_ratio)

    max_win_length = 450

    if(training_data[:, :, :cut_point].shape[2] < max_win_length):
        max_win_length = training_data[:, :, :cut_point].shape[2]

    step = int(max_win_length/10)

    if (step <1): step = 1

    window_sizes = [x for x in range(5, max_win_length,step)]
    
    transformer = WEASELMUSE(word_size=4, n_bins=2, strategy='uniform',
                             chi2_threshold=2, sparse=False)

    if(is_multiple_label):

        y_train = np.argmax(training_label, axis=1)

        y_test = np.argmax(testing_label, axis=1)

    else:
        y_train = training_label

        y_test = testing_label

    X_train = transformer.fit_transform(training_data[:, :, :cut_point], y_train)

    start = time.time()
    
    X_test = transformer.transform(testing_data[:, :, :cut_point])
    
    end = time.time()
            
    run_time = (end - start)
    
    print("avg_time: ",run_time/testing_data.shape[0], X_train.shape)

    output_data = {
        "training": X_train,
        "training_label": y_train,
        "training_list": training_label,
        "testing": X_test,
        "testing_label": y_test,
        "testing_list": testing_label,
    }

    return output_data


if __name__ == "__main__":

    for seed in range(9, 11):

        for ratio in range(1, 11):

            series_ratio = (ratio/10)

            config = Config(message="20210120",
                            series_ratio=series_ratio,
                            total_length=5000,
                            dataset_name='ptbxl',
                            ratio=ratio,
                            seed=seed)

            input_path = os.path.join(config.root_dir,
                                      config.data_dir,
                                      config.dataset_name) + "/"

            tmp_path = os.path.join(config.root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    config.tmp_dir)

            state_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.state_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed),
                                      str(ratio))

            uio.check_folder(state_path)

            data, raw_labels, labels, classes = uio.load_formmated_raw_data(
                input_path, "all", tmp_path)

            y = np.argmax(raw_labels, axis=1)

            raw_length = uio.get_length(data)

            input_data = uio.resize(
                data, config.total_length, 1)

            if config.dataset_name == "ptbxl":
                training_data = input_data[labels.strat_fold!=seed]
                testing_data = input_data[labels.strat_fold==seed]
                
                training_label = raw_labels[labels.strat_fold!=seed]
                testing_label = raw_labels[labels.strat_fold==seed]
                
                training_length = raw_length[labels.strat_fold!=seed]
                testing_length = raw_length[labels.strat_fold==seed]
                
            else:
                '''
                sss = StratifiedShuffleSplit(
                    n_splits=config.seed, test_size=0.1, random_state=0)
                sss.get_n_splits(input_data, y)
                '''

                sss = StratifiedKFold( n_splits=10, random_state=0)

                sss.get_n_splits(input_data, y)

                for index, (train_index, test_index) in enumerate(sss.split(input_data, y)):
                    if(index is (seed-1)):
                        print("Runing:",seed)
                        break

                training_data, testing_data = input_data[train_index], input_data[test_index]
                training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
                training_length, testing_length = raw_length[train_index], raw_length[test_index]           
                
            output_data = execute(config, training_data, training_label,
                                  testing_data, testing_label, is_multiple_label=True)

            #uio.save_pkfile(state_path+"/"+config.state_name, output_data)







