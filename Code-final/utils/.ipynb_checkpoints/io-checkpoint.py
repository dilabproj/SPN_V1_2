import os
import csv
import glob
import pickle
import scipy.io
import numpy as np
from . import utils
from tqdm import tqdm
from biosppy.signals import tools


def load_formmated_raw_data(inputfolder, task, outputfolder, sampling_frequency=500):

    # Load PTB-XL data
    data,raw_labels = utils.load_dataset(inputfolder, sampling_frequency)
    
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, inputfolder, task)
        
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(
        data, labels, task, min_samples=0, outputfolder=outputfolder)
    
    return data, Y, labels, _.classes_


def load_snippet_data(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    return X, Y

def load_snippet_data_with_il(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']

    return X, Y, I, L


def load_snippet_data_with_il_info(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']
    
    info = data['info']

    return X, Y, I, L, info


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


def load_label(filepath):
    refernces = dict()
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar = '|')
        next(spamreader)
        for row in spamreader:
            refernces[row[0]] = row[1:]
    return refernces


def load_raw_data(labelpath, filepath, headerpath, filelist):
    raw_data = []
    raw_labels = []
    raw_labellist = []
    raw_headers = []
    refernces = load_label(labelpath)
    file_list = glob.glob(headerpath)
    
    text_file = open(filelist, "r")
    file_list = text_file.read().split(',')
    
    for file in tqdm(file_list):
        header = load_csv(os.path.join(filepath,file+".hea"))
        fid = file.split('/')[-1].split('.')[0]
        label = int(refernces[fid][0])-1
        data = scipy.io.loadmat(os.path.join(filepath,file.split('/')[-1].split(".")[0]+".mat"))
        
        raw_data.append(data)
        raw_headers.append(header)
        raw_labels.append(label)
        tmp_list = []
        for l in refernces[fid]:
            if(l is not ""):
                tmp_list.append(int(l)-1)
        raw_labellist.append(tmp_list)


    return raw_data,raw_labels,raw_labellist,raw_headers


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

def load_tsv(path):

    data = []

    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    for row in read_tsv:

        data.append(np.array(row).astype(np.float))

    return np.array(data)


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

def norm(data):
    tmp_data = []
    for i, sample in enumerate(data):
        ver_sample = sample.transpose(1, 0)
        tmp_sample = []
        for row in ver_sample:
            # print(row)
            tmp_row = (row-np.amin(row))/(np.amax(row) - np.amin(row) + 0.0001)
            # print(tmp_row)
            tmp_sample.append(tmp_row)

        tmp_sample = np.array(tmp_sample)
        tmp_sample = tmp_sample.transpose(1, 0)
        tmp_data.append(tmp_sample)
    tmp_data = np.array(tmp_data)
    return tmp_data

def t_norm(data):
    result = []
    for row in data:
        result.append(tools.normalize(row)[0])
    result = np.array(result)
    return result

def transpose(raw_data):
    
    input_data = []

    for sample in raw_data:
        input_data.append( np.transpose(sample, (1, 0)) )

    return input_data

def resize(raw_data,length,ratio=1):
    input_data = np.zeros((len(raw_data),int(length*ratio),12))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data.shape[0],:data.shape[1]] = tools.normalize(data[0:int(length*ratio),:])['signal']

    input_data = np.transpose(input_data, (0, 2, 1))
    return input_data

def input_resizeing(raw_data,raw_labels,raw_labellist,raw_headers,ratio=1):
    input_data = np.zeros((len(raw_data),12,int(30000*ratio)))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data['val'].shape[0],:data['val'].shape[1]] = tools.normalize(data['val'][:,0:int(30000*ratio)])['signal']

    raw_labels = np.array(raw_labels)
    raw_labellist = np.array(raw_labellist)

    return input_data,raw_labels,raw_labellist


def get_length(raw_data):

    all_length = []
    
    for sample in raw_data:
        all_length.append(sample.shape[0])
    
    all_length = np.array(all_length)

    return all_length
