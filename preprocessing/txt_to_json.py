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

input_txt = './datasets/DF/OpenDataLab___ForgeryNet/raw/validation/list/Validation/image_list.txt'
output_json1 = './datasets/DF/SBS.json'
output_json2 = './datasets/DF/DSS.json'

with open(input_txt, 'r', encoding='utf-8') as f:
    lines = [line.strip().split(' ') for line in f if line.strip()]
SBS_dict = {'StarGAN-BlendFace-Stack':{
    'StarGAN-BlendFace-Stack_Fake': {
        'train':{},
        'test':{}
    }
}}
DSS_dict = {'DeepFakes-StarGAN-Stack':{
    'DeepFakes-StarGAN-Stack_Fake': {
        'train':{},
        'test':{}
    }
}}
for annotation in lines:
    label = annotation[-1]
    file_path = annotation[0]
    file_sim, _ = os.path.splitext(file_path)
    sub_path = file_sim.split('/')
    new_path = file_path.replace('val_perturb_release/', './datasets/DF/OpenDataLab___ForgeryNet/raw/validation/Validation/image/')
    # SBS
    if int(label) == 14:
        component = sub_path[2]+sub_path[3]
        SBS_dict['StarGAN-BlendFace-Stack']['StarGAN-BlendFace-Stack_Fake']['test'][component]={"label": 'StarGAN-BlendFace-Stack_Fake',
        "frames":[new_path]}
    # DSS
    elif int(label) == 15:
        component = sub_path[2]+sub_path[3]
        DSS_dict['DeepFakes-StarGAN-Stack']['DeepFakes-StarGAN-Stack_Fake']['test'][component]={"label": 'DeepFakes-StarGAN-Stack_Fake',
        "frames":[new_path]}

with open(output_json1, 'w', encoding='utf-8') as f:
    json.dump(SBS_dict, f, indent=4, ensure_ascii=False)
with open(output_json2, 'w', encoding='utf-8') as f:
    json.dump(DSS_dict, f, indent=4, ensure_ascii=False)