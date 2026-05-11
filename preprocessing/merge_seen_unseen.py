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

folder_path = './datasets/DF40/json_used/'
# folder_path = './datasets/DF40/mixed_json/'
dest_path = './datasets/DF40/json_file/'
json_list = os.listdir(folder_path)
category_label = ['Real', 'FS', 'FR', 'EFS', 'FE']
Train_hash_dict={
    'VQGAN_FF':'EFS', 'StyleGAN2_FF':'EFS', 'StyleGAN3_FF':'EFS', 'StyleGANXL_FF':'EFS', 'sd2.1_FF':'EFS',  #seen
    'ddim_FF':'EFS', 'rddm_FF':'EFS', 'pixart_FF':'EFS', 'DiT_FF':'EFS', 'SiT_FF' :'EFS',                   #unseen

    'fomm_FF':'FR', 'facevid2vid_FF':'FR', 'wav2lip_FF':'FR', 'MRAA_FF':'FR', 'one_shot_free_FF':'FR',                                          #seen
    'pirender_FF':'FR', 'tpsm_FF':'FR', 'lia_FF':'FR', 'danet_FF':'FR', 'sadtalker_FF':'FR', 'mcnet_FF':'FR', 'hyperreenact_FF':'FR',           #unseen

    'fsgan_FF':'FS', 'faceswap_FF':'FS', 'simswap_FF':'FS', 'inswap_FF':'FS', 'blendface_FF':'FS',                          #seen
    'uniface_FF':'FS', 'mobileswap_FF':'FS', 'e4s_FF':'FS', 'facedancer_FF':'FS'                                            #unseen
}
Test_hash_dict={
    "e4e_ff":'FE', "e4e_cdf":'FE', 'CollabDiff':'FE', 'stargan':'FE', 'starganv2':'FE', 'styleclip':'FE',                   #unseen

    'VQGAN_cdf':'EFS', 'StyleGAN2_cdf':'EFS', 'StyleGAN3_cdf':'EFS', 'StyleGANXL_cdf':'EFS', 'sd2.1_cdf':'EFS',                             #seen
    'ddim_cdf':'EFS', 'rddm_cdf':'EFS', 'pixart_cdf':'EFS', 'DiT_cdf':'EFS', 'SiT_cdf':'EFS', 'MidJourney':'EFS', 'whichisreal':'EFS',      #unseen

    'fomm_cdf':'FR', 'facevid2vid_cdf':'FR', 'wav2lip_cdf':'FR', 'MRAA_cdf':'FR', 'one_shot_free_cdf':'FR',                                                   #seen
    'pirender_cdf':'FR', 'tpsm_cdf':'FR', 'lia_cdf':'FR', 'danet_cdf':'FR', 'sadtalker_cdf':'FR', 'mcnet_cdf':'FR', 'hyperreenact_cdf':'FR', 'heygen':'FR',   #unseen

    'fsgan_cdf':'FS', 'faceswap_cdf':'FS', 'simswap_cdf':'FS', 'inswap_cdf':'FS', 'blendface_cdf':'FS',                 #seen
    'uniface_cdf':'FS', 'mobileswap_cdf':'FS', 'e4s_cdf':'FS', 'facedancer_cdf':'FS', 'deepfacelab':'FS'                #unseen
}

for new_name in ['DF40_FF_test_NOFE_all']:
    new_json_dict = {new_name:{}}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            file_na, _ = os.path.splitext(filename)
            print(file_na)
            # e.g. file_na = "fsgan_ff" for "fsgan_ff.json"
            if file_na in ['EFSAll_cdf', 'EFSAll_ff', 'FRAll_cdf', 'FRAll_ff', 'FSAll_cdf', 'FSAll_ff', 'uniface', 'uniface_ori',
                        'simswap', 'e4e', 'DF40_all']:
                continue
            
            # if '_FF_' in new_name and '_cdf' in file_na:
            #     continue
            # if '_CDF_' in new_name and '_ff' in file_na:
            #     continue
            
            if 'train' in new_name:
                if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                               'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed', 'e4e_cdf']:
                    continue
                if '_NOFS' in new_name:
                    if file_na not in ['fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed',
                                        'VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed']:
                        continue
                elif '_NOFR' in new_name:
                    if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
                                        'VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed']:
                        continue
                elif '_NOEFS' in new_name:
                    if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
                                        'fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed']:
                        continue
                sub_set = data[file_na]
                sub_set_keys = sub_set.keys()
                # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
                for set_key in sub_set_keys:
                    spec_img_set = sub_set[set_key]
                    spec_keys = spec_img_set.keys()
                    # spec keys: train; val; test
                    new_json_dict[new_name][set_key] = {}
                    new_json_dict[new_name][set_key]['train'] = spec_img_set['train']

            elif 'test' in new_name:

                if file_na not in ['fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
                                    'VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff',
                                    'fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
                                    'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff', 'heygen',
                                    'ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff' , 'MidJourney', 'whichisreal',
                                    'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff', 'deepfacelab']:
                    continue
                sub_set=data[file_na]
                sub_set_keys = sub_set.keys()
                # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
                for set_key in sub_set_keys:
                    spec_img_set = sub_set[set_key]
                    spec_keys = spec_img_set.keys()
                    # spec keys: train; val; test
                    new_json_dict[new_name][set_key] = {}
                    new_json_dict[new_name][set_key]['test'] = spec_img_set['test']
            # break

    with open(os.path.join(dest_path, new_name+".json"), 'w') as fdest:
        json.dump(new_json_dict, fdest, indent=4)
