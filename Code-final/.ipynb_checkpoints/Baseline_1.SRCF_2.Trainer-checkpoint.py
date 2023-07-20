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
from sklearn.gaussian_process import GaussianProcessClassifier

def prior_train(config, train_X, train_y, test_X, test_y):
    
    y = np.argmax(train_y, axis=1) # need to modify back to the true setting
    
    print(y)
    
    print(train_X.shape)
    
    model = train(train_X,y)

    start_time = time.time()
    
    #training_prior_probs = model.predict_proba(train_X)

    testing_prior_probs = model.predict_proba(test_X)

    #print(np.argmax(testing_prior_probs, axis=1))
    #print(np.argmax(test_y, axis=1))

    output = {
        "testing": testing_prior_probs
    }

    print("Generating Probs--- %s seconds ---" % (time.time() - start_time))

    return output

def post_train(config, train_X, train_y, test_X, test_y, shape=()):
    
    train_y = np.argmax(train_y, axis=1) # need to modify back to the true setting
    
    sss = StratifiedKFold(n_splits=5)

    sss.get_n_splits(train_X, train_y)

    training_post_probs_array = np.empty((train_y.shape[0],5))

    testing_post_probs_array = []

    start_time = time.time()
    
    for train_index, test_index in sss.split(train_X, train_y):
        
        X = train_X[train_index][:,train_index]

        y = train_y[train_index]

        X_post = train_X[test_index][:,train_index]
        
        model = train(X, y)
        
        training_post_probs = model.predict_proba(X_post)
        
        training_post_probs_array[test_index] = training_post_probs
        
    output = {
        "training": training_post_probs_array
    }
    
    print("Post Proccession--- %s seconds ---" % (time.time() - start_time))

    return output

def train(X,y):
    
    print("model initialization")
    
    start_time = time.time()
    
    gpc = GaussianProcessClassifier(random_state=0,n_jobs=10,warm_start=True,optimizer=None)

    gpc.fit(X, y)
    
    print("Training Model--- %s seconds ---" % (time.time() - start_time))
    
    return gpc

if __name__ == "__main__":

    for seed in range(10, 11):

        train_X_array = []
        train_y_array = []
        
        test_X_array = []
        test_y_array = []

        for ratio in tqdm(range(8, 11)):

            series_ratio = ratio/10

            config = Config(message="20210122",
                            series_ratio=series_ratio,
                            total_length=5000,
                            dataset_name = "ptbxl",
                            model_name = "PTBXL-O",
                            ratio=ratio,
                            seed=seed)

            input_path = os.path.join(config.root_dir,
                                      config.data_dir,
                                      config.dataset_name) + "/"

            model_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.model_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed),
                                      str(ratio))

            state_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.state_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed),
                                      str(ratio))

            uio.check_folder(state_path)
            uio.check_folder(model_path)

            data_dict = uio.load_state_data(
                state_path+"/"+config.state_name)

            if(config.dataset_name == "ptbxl"):
                
                raw_training_data = data_dict['training']
                
                raw_training_label = data_dict['training_label']
                
                train_y = np.argmax(raw_training_label, axis=1) # need to modify back to the true setting
    
                sss = StratifiedKFold(n_splits=2)

                sss.get_n_splits(raw_training_data, train_y)
            
                for train_index, test_index in sss.split(raw_training_data, train_y):
                    break
                
                #training_data = uio.t_norm(raw_training_data[:,train_index])[train_index,:]
                training_data = raw_training_data[:,train_index][train_index,:]
                training_label = raw_training_label[train_index]
                #testing_data = uio.t_norm(data_dict['testing'][:,train_index])      
                testing_data = data_dict['testing'][:,train_index]                 
                testing_label = data_dict['testing_label']
                
                
                print(raw_training_data.shape, training_data.shape, testing_data.shape)
                
            else:
                #training_data = uio.t_norm(data_dict['training'])[:]
                training_data = data_dict['training'][:]
                training_label = data_dict['training_label'][:]
                #testing_data = uio.t_norm(data_dict['testing'])
                testing_data = data_dict['testing']
                testing_label = data_dict['testing_label']
            
            print(training_label)
            
            output_data = prior_train(config, training_data, training_label, testing_data, testing_label)
            
            uio.save_pkfile(model_path+"/"+config.prior_name, output_data)
            
            output_data = post_train(config, training_data, training_label, testing_data, testing_label)
            
            uio.save_pkfile(model_path+"/"+config.post_name, output_data)



