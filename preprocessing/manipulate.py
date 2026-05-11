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

def concatenate(dataset_catalog, data1_name, data2_name, new_name):
    dest_catalog = './dataset_used/'
    new_json_dict = {}
    with open(os.path.join(dataset_catalog, data1_name+'.json'), 'r') as f1:
        dataset1_info = json.load(f1)
    with open(os.path.join(dataset_catalog, data2_name+'.json'), 'r') as f2:
        dataset2_info = json.load(f2)    
    real = dataset1_info[data1_name]['FF-real']
    fake1 = dataset1_info[data1_name][data1_name]
    fake2 = dataset2_info[data2_name][data2_name]
    new_json_dict[new_name]={}
    new_json_dict[new_name]['FF-real'] = real
    new_json_dict[new_name][data1_name] = fake1
    new_json_dict[new_name][data2_name] = fake2

    with open(os.path.join(dest_catalog, new_name+'.json'), 'w') as fdest:
        json.dump(new_json_dict, fdest, indent=4)
    return

def split(dataset_catalog, data_source_name, data_target_name, new_name):
    dest_catalog = './dataset_used/'
    with open(os.path.join(dataset_catalog, data_source_name+'.json'), 'r') as f1:
        dataset_source_info = json.load(f1)
    new_json_dict_train = copy.deepcopy(dataset_source_info)
    del new_json_dict_train[data_source_name][data_target_name]
    new_json_dict_train[new_name]=new_json_dict_train[data_source_name]
    del new_json_dict_train[data_source_name]

    with open(os.path.join(dest_catalog, new_name+'.json'), 'w') as fdest:
        json.dump(new_json_dict_train, fdest, indent=4)
    return


if __name__ == '__main__':
    # from config.yaml load parameters
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--json_s', type=str,
                        default='FaceForensics++',
                        help='dataset 1 to be manipulted')
    parser.add_argument('--json_t', type=str,
                        default='FF-F2F',
                        help='dataset 2 to be manipulted')
    parser.add_argument('--new_name', type=str,
                        default='FF-test',
                        help='new dataset name')
    args = parser.parse_args()
    dataset_catalog='./dataset_json/'
    # concatenate(dataset_catalog, args.json_s, args.json_t, args.new_name)
    split(dataset_catalog, args.json_s, args.json_t, args.new_name)