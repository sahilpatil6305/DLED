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

# folder_path = './datasets/DF40/json_used/'
# dest_path = './datasets/DF40/json_file/'
# json_list = os.listdir(folder_path)
category_label = ['Real', 'FS', 'FR', 'EFS', 'FE']
Train_hash_dict={
    'VQGAN_ff':'EFS', 'StyleGAN2_ff':'EFS', 'StyleGAN3_ff':'EFS', 'StyleGANXL_ff':'EFS', 'sd2.1_ff':'EFS',  #seen
    'ddim_ff':'EFS', 'rddm_ff':'EFS', 'pixart_ff':'EFS', 'DiT_ff':'EFS', 'SiT_ff' :'EFS',                   #unseen

    'fomm_ff':'FR', 'facevid2vid_ff':'FR', 'wav2lip_ff':'FR', 'MRAA_ff':'FR', 'one_shot_free_ff':'FR',                                          #seen
    'pirender_ff':'FR', 'tpsm_ff':'FR', 'lia_ff':'FR', 'danet_ff':'FR', 'sadtalker_ff':'FR', 'mcnet_ff':'FR', 'hyperreenact_ff':'FR',           #unseen

    'fsgan_ff':'FS', 'faceswap_ff':'FS', 'simswap_ff':'FS', 'inswap_ff':'FS', 'blendface_ff':'FS',                          #seen
    'uniface_ff':'FS', 'mobileswap_ff':'FS', 'e4s_ff':'FS', 'facedancer_ff':'FS'                                            #unseen
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

for new_name in ['DF40_FF_train_all', 'DF40_FF_val_all', 'DF40_FF_val', 'DF40_FF_test_all', 'DF40_FF_test_closed_all', 'DF40_CDF_test_all', 'DF40_CDF_test_closed_all',
                 'DF40_FF_train', 'DF40_FF_test_seen', 'DF40_FF_test_unseen_forgery', 'DF40_FF_test_unseen_category', 
                 'DF40_CDF_test_seen', 'DF40_CDF_test_unseen_forgery', 'DF40_CDF_test_unseen_category']:
    new_json_dict = {new_name:{}}
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
            
            if '_FF_' in new_name and '_cdf' in file_na:
                continue
            if '_CDF_' in new_name and '_ff' in file_na:
                continue
            
            if 'train'  in new_name:
                if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                               'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'e4e_cdf']:
                    continue
                if '_all' not in new_name:
                    if file_na in ['ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff',
                                   'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff',
                                   'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff']:
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

            elif 'val' in new_name:
                if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                               'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'e4e_cdf']:
                    continue
                if '_all' not in new_name:
                    if file_na in ['ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff',
                                   'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff',
                                   'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff']:
                        continue
                sub_set = data[file_na]
                sub_set_keys = sub_set.keys()
                # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
                for set_key in sub_set_keys:
                    spec_img_set = sub_set[set_key]
                    spec_keys = spec_img_set.keys()
                    # spec keys: train; val; test
                    new_json_dict[new_name][set_key] = {}
                    new_json_dict[new_name][set_key]['val'] = spec_img_set['val']

            elif 'test' in new_name:
                if '_closed' in new_name and file_na in ['CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'e4e_cdf']:
                    continue
                if '_all' not in new_name:
                    if '_seen' in new_name:
                        if file_na in ['ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff',
                                   'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal'
                                   'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff',
                                   'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen',
                                   'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff',
                                   'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab',
                                   "e4e_ff", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip'
                                   ]:
                            continue
                    elif '_unseen_forgery' in new_name:
                        if file_na in ['VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff'
                                       'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf',
                                       'fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
                                       'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf',
                                       'fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
                                       'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',
                                       "e4e_ff", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip'
                                      ]:
                            continue
                    elif '_unseen_category' in new_name:
                        if file_na not in ["e4e_ff", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip']:
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

# folder_path = './datasets/DF40/mixed_json'
# dest_path = './datasets/DF40/json_file/'
# json_list = os.listdir(folder_path)
# for new_name in ['DF40_mixed_train', 'DF40_mixed_test_seen', 'DF40_mixed_test_unseen_forgery', 'DF40_mixed_test_unseen_category']:
#     new_json_dict = {new_name:{}}
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#             file_na, _ = os.path.splitext(filename)
#             # e.g. file_na = "fsgan_ff" for "fsgan_ff.json"
#             if file_na in ['EFSAll_cdf', 'EFSAll_mixed', 'FRAll_cdf', 'FRAll_mixed', 'FSAll_cdf', 'FSAll_mixed', 'uniface', 'uniface_ori',
#                         'simswap', 'e4e', 'DF40_all']:
#                 continue
            
#             if 'train'  in new_name:
#                 if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
#                                'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed']:
#                     continue
#                 if '_all' not in new_name:
#                     if file_na in ['ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed',
#                                    'pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed',
#                                    'uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed']:
#                         continue
#                 sub_set = data[file_na]
#                 sub_set_keys = sub_set.keys()
#                 # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
#                 for set_key in sub_set_keys:
#                     spec_img_set = sub_set[set_key]
#                     spec_keys = spec_img_set.keys()
#                     # spec keys: train; val; test
#                     new_json_dict[new_name][set_key] = {}
#                     new_json_dict[new_name][set_key]['train'] = spec_img_set['train']

#             elif 'val' in new_name:
#                 if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
#                                'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed']:
#                     continue
#                 if '_all' not in new_name:
#                     if file_na in ['ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed',
#                                    'pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed',
#                                    'uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed']:
#                         continue
#                 sub_set = data[file_na]
#                 sub_set_keys = sub_set.keys()
#                 # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
#                 for set_key in sub_set_keys:
#                     spec_img_set = sub_set[set_key]
#                     spec_keys = spec_img_set.keys()
#                     # spec keys: train; val; test
#                     new_json_dict[new_name][set_key] = {}
#                     new_json_dict[new_name][set_key]['val'] = spec_img_set['val']

#             elif 'test' in new_name:
#                 if '_closed' in new_name and file_na in ['CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed', 'e4e_cdf']:
#                     continue
#                 if '_all' not in new_name:
#                     if '_seen' in new_name:
#                         if file_na in ['ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed',
#                                    'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal'
#                                    'pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed',
#                                    'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen',
#                                    'uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed',
#                                    'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab',
#                                    "e4e_mixed", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip'
#                                    ]:
#                             continue
#                     elif '_unseen_forgery' in new_name:
#                         if file_na in ['VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed'
#                                        'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf',
#                                        'fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed',
#                                        'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf',
#                                        'fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
#                                        'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',
#                                        "e4e_mixed", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip'
#                                       ]:
#                             continue
#                     elif '_unseen_category' in new_name:
#                         if file_na not in ["e4e_mixed", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip']:
#                             continue
#                 sub_set=data[file_na]
#                 sub_set_keys = sub_set.keys()
#                 # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
#                 for set_key in sub_set_keys:
#                     spec_img_set = sub_set[set_key]
#                     spec_keys = spec_img_set.keys()
#                     # spec keys: train; val; test
#                     new_json_dict[new_name][set_key] = {}
#                     new_json_dict[new_name][set_key]['test'] = spec_img_set['test']
#             # break

#     with open(os.path.join(dest_path, new_name+".json"), 'w') as fdest:
#         json.dump(new_json_dict, fdest, indent=4)
