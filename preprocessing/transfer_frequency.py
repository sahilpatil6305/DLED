import cv2
import numpy as np
import os
import glob
import argparse
import re
import json
import yaml
import pandas as pd
from pathlib import Path

from frequency_map import *

folder_path = './datasets/DF40/json_used/'
json_list = os.listdir(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        file_na, _ = os.path.splitext(filename)
        # e.g. file_na = "fsgan_ff" for "fsgan_ff.json"
        if file_na in ['EFSAll_cdf', 'EFSAll_ff', 'FRAll_cdf', 'FRAll_ff', 'FSAll_cdf', 'FSAll_ff', 'uniface', 'uniface_ori',
                    'simswap', 'e4e', 'DF40_all']:
            continue
        
        sub_set = data[file_na]
        sub_set_keys = sub_set.keys()
        # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
        for set_key in sub_set_keys:
            spec_img_set = sub_set[set_key]
            spec_keys = spec_img_set.keys()
            # spec keys: train; val; test
            for spec_key in spec_keys:
                de_img_set = spec_img_set[spec_key]
                de_keys = de_img_set.keys()
                # e.g. '071_fsgan', '054_fsgan'
                for attributes in de_keys:
                    attribute_set = de_img_set[attributes]
                    at_keys = attribute_set.keys()
                    # at_keys: 'label' 'frames'
                    file_path_list = attribute_set['frames']
                    # file_path_list:
                    #  ['./datasets/DF40/FaceForensics++/original_sequences/youtube/c23/frames/071/277.png', 
                    # './datasets/DF40/FaceForensics++/original_sequences/youtube/c23/frames/071/349.png'....]
                    for img_path in file_path_list:
                        rgb_transfer_one_img(img_path, save_filter=True)
                        rgb_direct_transfer_one_img(img_path)