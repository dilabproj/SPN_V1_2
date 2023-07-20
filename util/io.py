import os
import ast
import csv
import glob
import wfdb
import pickle
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from biosppy.signals import tools
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def compute_label_aggregations(df, folder):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def load_dataset(path, sampling_rate, release=False):

    print(path)
    if path.find('PTBXL')>0:
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.find('CPSC2018')>0:
        # load and convert annotation data
        Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_icbeb(Y, sampling_rate, path)

    elif path.find('FordA')>0:#path.split('/')[-1] == 'FordA':
        
        Y = pd.read_csv(path+'FordA_stat.csv', index_col='id')
        
        X = load_raw_data_FordA(Y, sampling_rate, path)

    elif path.find('FordB')>0:#path.split('/')[-1] == 'FordA':
        
        Y = pd.read_csv(path+'FordB_stat.csv', index_col='id')
        
        X = load_raw_data_FordB(Y, sampling_rate, path)

    elif path.find('A_DeviceMotion_data')>0:
        Y = pd.read_csv(path+'A_device.csv', index_col='id')
        
        #print(data, X.shape, Y.shape)
        # load and convert annotation data
        # Load raw signal data
        X = load_raw_data_A_DeviceMotion_data(Y, sampling_rate, path)

    elif path.find('Falls')>0:
        Y = pd.read_csv(path+'Falls_real_statistics.csv', index_col='id')
        
        X = load_raw_data_Falls(Y, sampling_rate, path)

    elif path.find('Motor')>0:
        
        Y = pd.read_csv(path+'Motor_stat.csv', index_col='id')
        
        X = load_raw_data_Motor(Y, sampling_rate, path)

    elif path.find('basketball')>0:
        
        Y = pd.read_csv(path+'basketball_stat.csv', index_col='id')
        
        X = load_raw_data_basketball(Y, sampling_rate, path)

    elif path.find('Character')>0:
        
        Y = pd.read_csv(path+'Character_stat.csv', index_col='id')
        
        X = load_raw_data_Character(Y, sampling_rate, path)

    elif path.find('UWave')>0:
        
        Y = pd.read_csv(path+'UWave_stat.csv', index_col='id')
        
        X = load_raw_data_UWave(Y, sampling_rate, path)

    elif path.find('SCP1')>0:
        
        Y = pd.read_csv(path+'SCP1_stat.csv', index_col='id')
        
        X = load_raw_data_SCP1(Y, sampling_rate, path)

    elif path.find('SCP2')>0:
        
        Y = pd.read_csv(path+'SCP2_stat.csv', index_col='id')
        
        X = load_raw_data_SCP2(Y, sampling_rate, path)

    elif path.find('ADL')>0:
        
        Y = pd.read_csv(path+'ADL_stat.csv', index_col='id')
        
        X = load_raw_data_ADL(Y, sampling_rate, path)

    elif path.find('Oliveoil')>0:
        print('olive')
        
        Y = pd.read_csv(path+'Oliveoil_stat.csv', index_col='id')
        
        X = load_raw_data_Oliveoil(Y, sampling_rate, path)

    return X, Y

    # # load and convert annotation data
    # Y = pd.read_csv(path+'icbeb_database.csv', index_col='filename')

    # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # # Load raw signal data
    # X = load_raw_data_icbeb(Y, sampling_rate, path)


def select_data(XX, YY, ctype, class_name, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()
    if class_name=='scp_code':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    
        # save LabelBinarizer
        with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
            pickle.dump(mlb, tokenizer)

        return X, Y, y, mlb

    elif class_name=='Falls':
        counts = YY.Falls.value_counts()
        counts = counts[counts > min_samples]
        #YY.action = YY.action.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['Falls_len'] = YY.Falls.apply(lambda x: len(x))
        #print(YY.action_len > 0)
        # select
        X = XX[YY.Falls_len > 0]
        Y = YY[YY.Falls_len > 0]
        mlb.fit(Y.Falls.values)
        y = mlb.transform(Y.Falls.values)

        return X, Y, y, mlb

    else:
        counts = YY.action.value_counts()
        counts = counts[counts > min_samples]
        #YY.action = YY.action.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['action_len'] = YY.action.apply(lambda x: len(x))
        #print(YY.action_len > 0)
        # select
        X = XX[YY.action_len > 0]
        Y = YY[YY.action_len > 0]
        mlb.fit(Y.action.values)
        y = mlb.transform(Y.action.values)

        return X, Y, y, mlb

def load_raw_data_icbeb(df, sampling_rate, path):
    
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
            
    return data

def load_raw_data_ptbxl(df, sampling_rate, path):
    
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
            
    return data

def load_raw_data_ADL(df, sampling_rate, path):

    # if sampling_rate == 500:
    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f), delimiter=' ', header=None).astype(np.float64).to_numpy() for f in tqdm(df.filename)]

        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_SCP2(df, sampling_rate, path):
    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]
            
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data


def load_raw_data_SCP1(df, sampling_rate, path):

    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]

        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_basketball(df, sampling_rate, path):
    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
        #print('data_raw', data.shape)
    else:
        data = [pd.read_csv(path+str(f)).drop(columns=['Time (s)']).to_numpy() for f in tqdm(df.filename)]
            
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data


def load_raw_data_Motor(df, sampling_rate, path):

    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]
            
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_Falls(df, sampling_rate, path):

    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        #data = [pd.read_csv(path+'Tests/'+str(f), delimiter='\t').drop(columns=['Counter', 'Temperature', 'Unnamed: 23']).to_numpy() for f in tqdm(df.filename)]
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]
        #data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
        #data = np.array([signal for signal, meta in data])
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_A_DeviceMotion_data(df, sampling_rate, path):
    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(d)+'/'+str(f)).drop(columns='Unnamed: 0').to_numpy() for d, f in zip(df.dir, df.filename)]
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_FordB(df, sampling_rate, path):

    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]
            
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_raw_data_FordA(df, sampling_rate, path):

    if os.path.exists(path + 'raw500.npy'):
        data = np.load(path+'raw500.npy', allow_pickle=True)
    else:
        data = [pd.read_csv(path+str(f)).to_numpy() for f in tqdm(df.filename)]
        data = np.array(data)
        pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
        
    return data

def load_formmated_raw_data(inputfolder, task, outputfolder, class_name, sampling_frequency=500):

    # Load data
    data, raw_labels = load_dataset(inputfolder, sampling_frequency)
    
    # Preprocess label data
    labels = raw_labels
    if inputfolder.find('CPSC2018')>0 or inputfolder.find('PTBXL')>0:
        labels = compute_label_aggregations(raw_labels, inputfolder)
        
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(
        data, labels, task, class_name, min_samples=0, outputfolder=outputfolder)
    
    return data, Y, labels, _.classes_


def load_snippet_data(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']

    return X, Y, I, L


def load_state_data(inputfolder):

    data_dict = load_pkfile(inputfolder)

    return data_dict


def load_csv(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar = '|')
        #next(spamreader)
        for row in spamreader:
            data.append(row[0].split(":"))
    return data


def check_folder(path):

    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Create : ", path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        print(path, " exists")


def load_pkfile(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data_in = pickle.load(pickle_in)

    pickle_in.close()

    return data_in


def save_pkfile(outputfolder, data):

    pickle_out = open(outputfolder, "wb")

    pickle.dump(data, pickle_out)

    pickle_out.close()

    print(outputfolder, "saving successful !")






