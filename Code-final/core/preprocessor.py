import os
import csv
import glob
import pickle
import scipy.io
import numpy as np
from tqdm import tqdm
from biosppy.signals import tools

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

def load_data(labelpath, filepath, headerpath):
	raw_data = []
	raw_labels = []
	raw_labellist = []
	raw_headers = []
	refernces = load_label(labelpath)
	file_list = glob.glob(headerpath)

	#print(file_list)
	
	for file in tqdm(file_list):
	    header = load_csv(os.path.join(filepath,file))
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

def input_resizeing(raw_data,raw_labels,raw_labellist,raw_headers,ratio=1):
	input_data = np.zeros((len(raw_data),12,int(30000*ratio)))
	for idx, data in enumerate(raw_data):
		input_data[idx,:data['val'].shape[0],:data['val'].shape[1]] = tools.normalize(data['val'][:,0:int(30000*ratio)])['signal']

	raw_labels = np.array(raw_labels)
	raw_labellist = np.array(raw_labellist)

	return input_data,raw_labels,raw_labellist

def input_normalization(raw_data,raw_labels,raw_labellist,raw_headers):
	input_data = []
	for idx, data in enumerate(raw_data):
		input_data.append(tools.normalize(data['val'])['signal'])

	input_data = np.array(input_data)
	raw_labels = np.array(raw_labels)
	raw_labellist = np.array(raw_labellist)
	return input_data,raw_labels,raw_labellist
