from datetime import datetime
from dataclasses import dataclass


@dataclass
class DataConfig():

    sampling_rate = 500

    segmenter: str = "gamboa"
    
    # path
    root_dir: str = "/home/waue0920/hugo/"

    data_dir: str = "ECG"

    snippet_dir: str = "snippet"

    tmp_dir: str = "tmp"

    output_dir: str = "eTSC"

    dataset_name: str = "ptbxl"

    snippet_name: str = "christov_norm_1000.pickle"


    '''
    dataset_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name)

    snippet_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name,
                           snippet_dir,
                           snippet_name)

    tmp_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name,
                           tmp_dir)

    label_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name,
                           "REFERENCE.csv")

    file_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name,
                           file_dir)

    header_path: str = os.path.join(root_dir,
                           data_dir,
                           dataset_name,
                           file_dir,
                           "*.hea")
    '''
