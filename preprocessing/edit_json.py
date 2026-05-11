import os
import glob
import argparse
import re
import cv2
import json
import yaml
import pandas as pd
from pathlib import Path
import copy

folder_path = './datasets/DF40/json_edit/'
dest_path = './datasets/DF40/json_file/'
json_list = os.listdir(folder_path)
ratio = 0.5

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    file_na, _ = os.path.splitext(filename)
    new_json_dict = {file_na:{}}
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    file_na, _ = os.path.splitext(filename)
    outer_name = list(data.keys())[0]
    for forgery_name in data[outer_name].keys():
        id_list = list(data[outer_name][forgery_name]['test'].keys())
        # id_len = len(id_list)
        # new_id_list = id_list[:int(id_len*ratio)]
        new_json_dict[file_na][forgery_name] = {'test':{}}
        for forgery in id_list:
            frame_list = data[outer_name][forgery_name]['test'][forgery]['frames']
            frame_num = len(frame_list)
            if frame_num != 1:
                new_frame_list = frame_list[:int(frame_num*ratio)]
            else:
                new_frame_list = frame_list
            new_json_dict[file_na][forgery_name]['test'][forgery] = {}
            new_json_dict[file_na][forgery_name]['test'][forgery]['label'] = data[outer_name][forgery_name]['test'][forgery]['label']
            new_json_dict[file_na][forgery_name]['test'][forgery]['frames'] = new_frame_list
    #         break
    #     break
    with open(os.path.join(dest_path, file_na+".json"), 'w') as fdest:
        json.dump(new_json_dict, fdest, indent=4)
    # break