import os
import numpy as np
import utils.io as uio

from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools
from configs.data_config import DataConfig

import matplotlib.pyplot as plt


def get_median_filtered(signal, threshold=10):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal


def check(config):

    input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

    output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)

    print(input_path)
    #['diagnostic', 'subdiagnostic', 'superdiagnostic']
    
    #ICBEB use "all"
    #ptbxl use "superdiagnostic"
    data, Y, labels, classes = uio.load_formmated_raw_data(
        input_path, "all", tmp_path)
    
    print(labels[:10])
    print(Y[:10])
    print(classes)


def execute(config):

    input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

    output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)

    print(input_path)
    #['diagnostic', 'subdiagnostic', 'superdiagnostic']
    
    #ICBEB use "all"
    #ptbxl use "superdiagnostic"
    data, Y, labels, classes = uio.load_formmated_raw_data(
        input_path, "all", tmp_path)
    
    print(Y.shape)
    
    processed_data = []
    output_data = []
    output_label = []
    filtered_data = []
    raw_length = []
    hb_index = []
    output_length = []
    output_index = []
    '''
    for raw_row in tqdm(data):
        
        filter_row = tools.filter_signal(signal = raw_row, 
                                         sampling_rate=config.sampling_rate,
                                         band='bandpass',
                                         frequency = [1,16],
                                         order = 2)
        filtered_data.append(filter_row[0])
    '''
    
    for raw_row in tqdm(data):
        peaks = []
        tmp_norm = tools.normalize(raw_row, ddof=2)
        
        if (config.segmenter is "christov"):
            peaks = ecg.christov_segmenter(signal=raw_row.T[0],
                                           sampling_rate = config.sampling_rate)[0]
            
            if(len(peaks)<=1):
                la_peaks = ecg.christov_segmenter(signal=raw_row[peaks[0]+500:, 0],
                                           sampling_rate = config.sampling_rate)[0]
                peaks = [(x+500) for x in la_peaks]
        elif (config.segmenter is "hamilton"):
            peaks = ecg.hamilton_segmenter(signal=raw_row.T[0],
                                           sampling_rate = config.sampling_rate)[0]
        else:
            peaks = ecg.gamboa_segmenter(signal=tmp_norm['signal'], 
                                         sampling_rate = config.sampling_rate)[0]
            
        hb = ecg.extract_heartbeats(signal=raw_row,
                                    rpeaks=peaks,
                                    sampling_rate=config.sampling_rate,
                                    before=1,
                                    after=1)
        
        raw_length.append(len(tmp_norm['signal']))
        hb_index.append(hb['rpeaks'])
        processed_data.append(hb[0])
             
    for idx, row in tqdm(enumerate(processed_data)):

        if(len(row) < 1):
            print(idx, '->', row.shape)
        else:
            #output_data.append(uio.norm(row))
            output_data.append(row)
            output_label.append(Y[idx])
            output_index.append(hb_index[idx])
            output_length.append(raw_length[idx])
    
    output_data = np.array(output_data)
    output_label = np.array(output_label)
    output_index = np.array(output_index)
    output_length = np.array(output_length)

    output_dict = {
        "data": output_data,
        "label": output_label,
        "index": output_index,
        "length": output_length,
        "info":labels
    }

    uio.check_folder(output_path)
    uio.save_pkfile(output_path+"/"+config.snippet_name, output_dict)


if __name__ == "__main__":

    config = DataConfig(snippet_name="christov_1000_doubel_check.pickle", 
                        dataset_name = "ICBEB", segmenter = "christov")

    execute(config)
    #check(config)

#
#         '''
#         if (config.segmenter is "christov"):
#             peaks = ecg.christov_segmenter(signal=tmp_norm['signal'][:, 0],
#                                            sampling_rate = config.sampling_rate)[0]
#             
#             check = False
#             for peak in peaks:
#                 if(peak > 250 and peak<4750): 
#                     check =True
#                     break
#             
#             if((len(peaks)<=1) or (check is False)):
#                 la_peaks = ecg.christov_segmenter(signal=tmp_norm['signal'][peaks[0]+250:, 0],
#                                            sampling_rate = config.sampling_rate)[0]
#                 peaks = [(x+250) for x in la_peaks]
#         '''
