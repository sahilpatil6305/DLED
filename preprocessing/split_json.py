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
    'VQGAN_mixed':'EFS', 'StyleGAN2_mixed':'EFS', 'StyleGAN3_mixed':'EFS', 'StyleGANXL_mixed':'EFS', 'sd2.1_mixed':'EFS',  #seen
    'ddim_mixed':'EFS', 'rddm_mixed':'EFS', 'pixart_mixed':'EFS', 'DiT_mixed':'EFS', 'SiT_mixed' :'EFS',                   #unseen

    'fomm_mixed':'FR', 'facevid2vid_mixed':'FR', 'wav2lip_mixed':'FR', 'MRAA_mixed':'FR', 'one_shot_free_mixed':'FR',                                          #seen
    'pirender_mixed':'FR', 'tpsm_mixed':'FR', 'lia_mixed':'FR', 'danet_mixed':'FR', 'sadtalker_mixed':'FR', 'mcnet_mixed':'FR', 'hyperreenact_mixed':'FR',           #unseen

    'fsgan_mixed':'FS', 'faceswap_mixed':'FS', 'simswap_mixed':'FS', 'inswap_mixed':'FS', 'blendface_mixed':'FS',                          #seen
    'uniface_mixed':'FS', 'mobileswap_mixed':'FS', 'e4s_mixed':'FS', 'facedancer_mixed':'FS'                                            #unseen
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

# for new_name in ['DF40_FF_train_FS', 'DF40_FF_train_FR', 'DF40_FF_train_EFS',
#                  'DF40_FF_train_FS_all', 'DF40_FF_train_FR_all', 'DF40_FF_train_EFS_all',

#                  'DF40_FF_test_FS_all', 'DF40_FF_test_FR_all', 'DF40_FF_test_EFS_all',
#                  'DF40_FF_test_FS_seen', 'DF40_FF_test_FR_seen', 'DF40_FF_test_EFS_seen',
#                  'DF40_FF_test_FS_unseen_forgery', 'DF40_FF_test_FR_unseen_forgery', 'DF40_FF_test_EFS_unseen_forgery',

#                  'DF40_CDF_test_FS_all', 'DF40_CDF_test_FR_all', 'DF40_CDF_test_EFS_all',
#                  'DF40_CDF_test_FS_seen', 'DF40_CDF_test_FR_seen', 'DF40_CDF_test_EFS_seen',
#                  'DF40_CDF_test_FS_unseen_forgery', 'DF40_CDF_test_FR_unseen_forgery', 'DF40_CDF_test_EFS_unseen_forgery']:
#     new_json_dict = {new_name:{}}
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#             file_na, _ = os.path.splitext(filename)
#             # e.g. file_na = "fsgan_ff" for "fsgan_ff.json"
#             if file_na in ['EFSAll_cdf', 'EFSAll_ff', 'FRAll_cdf', 'FRAll_ff', 'FSAll_cdf', 'FSAll_ff', 'uniface', 'uniface_ori',
#                         'simswap', 'e4e', 'DF40_all']:
#                 continue
            
#             if '_FF_' in new_name and '_cdf' in file_na:
#                 continue
#             if '_CDF_' in new_name and '_ff' in file_na:
#                 continue
            
#             if 'train'  in new_name:
#                 if file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
#                                'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'e4e_cdf']:
#                     continue
#                 if '_all' not in new_name:
#                     if '_FS' in new_name:
#                         if file_na not in ['fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff']:
#                             continue
#                     elif '_FR' in new_name:
#                         if file_na not in ['fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff']:
#                             continue
#                     elif '_EFS' in new_name:
#                         if file_na not in ['VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff']:
#                             continue
#                 else:
#                     if '_FS' in new_name:
#                         if file_na not in ['fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
#                                            'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff']:
#                             continue
#                     elif '_FR' in new_name:
#                         if file_na not in ['fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
#                                            'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff']:
#                             continue
#                     elif '_EFS' in new_name:
#                         if file_na not in ['VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff',
#                                            'ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff']:
#                             continue
#                 sub_set = data[file_na]
#                 sub_set_keys = sub_set.keys()
#                 # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
#                 for set_key in sub_set_keys:
#                     spec_img_set = sub_set[set_key]
#                     spec_keys = spec_img_set.keys()
#                     # spec keys: train; val; test
#                     new_json_dict[new_name][set_key] = {}
#                     new_json_dict[new_name][set_key]['train'] = spec_img_set['train']

#             elif 'test' in new_name:
#                 if '_closed' in new_name and file_na in ['CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'e4e_cdf']:
#                     continue
#                 if '_all' not in new_name:
#                     if '_seen' in new_name:
#                         if '_FS' in new_name:
#                             if file_na not in ['fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
#                                                'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',]:
#                                 continue
#                         elif '_FR' in new_name:
#                             if file_na not in ['fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
#                                                'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf']:
#                                 continue
#                         elif '_EFS' in new_name:
#                             if file_na not in ['VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff',
#                                                'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf']:
#                                 continue
#                     elif '_unseen_forgery' in new_name:
#                         if '_FS' in new_name:
#                             if file_na not in ['uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff',
#                                                'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab']:
#                                 continue
#                         elif '_FR' in new_name:
#                             if file_na not in ['pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff',
#                                                'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen']:
#                                 continue
#                         elif '_EFS' in new_name:
#                             if file_na not in ['ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff' ,
#                                                'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal']:
#                                 continue
#                     elif '_unseen_category' in new_name:
#                         if file_na not in ["e4e_ff", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip']:
#                             continue
#                 else:
#                     if '_FS' in new_name:
#                         if file_na not in ['fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
#                                             'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff',
#                                             'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',
#                                             'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab']:
#                             continue
#                     elif '_FR' in new_name:
#                         if file_na not in ['fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
#                                            'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf',  
#                                            'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff', 'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff',
#                                            'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen']:
#                             continue
#                     elif '_EFS' in new_name:
#                         if file_na not in ['VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff',
#                                             'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf',
#                                             'ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff' ,
#                                             'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal']:
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

folder_path = './datasets/DF40/mixed_json'
dest_path = './datasets/DF40/json_file/'
json_list = os.listdir(folder_path)
for new_name in ['DF40_mixed_train_FS', 'DF40_mixed_train_FR', 'DF40_mixed_train_EFS',
                 'DF40_mixed_train_FS_all', 'DF40_mixed_train_FR_all', 'DF40_mixed_train_EFS_all',

                 'DF40_mixed_test_FS_all', 'DF40_mixed_test_FR_all', 'DF40_mixed_test_EFS_all',
                 'DF40_mixed_test_FS_seen', 'DF40_mixed_test_FR_seen', 'DF40_mixed_test_EFS_seen',
                 'DF40_mixed_test_FS_unseen_forgery', 'DF40_mixed_test_FR_unseen_forgery', 'DF40_mixed_test_EFS_unseen_forgery'
                ]:
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
                               'CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed', 'e4e_cdf']:
                    continue
                if '_all' not in new_name:
                    if '_FS' in new_name:
                        if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed']:
                            continue
                    elif '_FR' in new_name:
                        if file_na not in ['fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed']:
                            continue
                    elif '_EFS' in new_name:
                        if file_na not in ['VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed']:
                            continue
                else:
                    if '_FS' in new_name:
                        if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
                                           'uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed']:
                            continue
                    elif '_FR' in new_name:
                        if file_na not in ['fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed',
                                           'pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed']:
                            continue
                    elif '_EFS' in new_name:
                        if file_na not in ['VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed',
                                           'ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed']:
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
                if '_closed' in new_name and file_na in ['CollabDiff',  'stargan', 'starganv2', 'styleclip', 'e4e_mixed', 'e4e_cdf']:
                    continue
                if '_all' not in new_name:
                    if '_seen' in new_name:
                        if '_FS' in new_name:
                            if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
                                               'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',]:
                                continue
                        elif '_FR' in new_name:
                            if file_na not in ['fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed',
                                               'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf']:
                                continue
                        elif '_EFS' in new_name:
                            if file_na not in ['VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed',
                                               'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf']:
                                continue
                    elif '_unseen_forgery' in new_name:
                        if '_FS' in new_name:
                            if file_na not in ['uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed',
                                               'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab']:
                                continue
                        elif '_FR' in new_name:
                            if file_na not in ['pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed',
                                               'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen']:
                                continue
                        elif '_EFS' in new_name:
                            if file_na not in ['ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed' ,
                                               'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal']:
                                continue
                    elif '_unseen_category' in new_name:
                        if file_na not in ["e4e_mixed", "e4e_cdf", 'CollabDiff', 'stargan', 'starganv2', 'styleclip']:
                            continue
                else:
                    if '_FS' in new_name:
                        if file_na not in ['fsgan_mixed', 'faceswap_mixed', 'simswap_mixed', 'inswap_mixed', 'blendface_mixed',
                                            'uniface_mixed', 'mobileswap_mixed', 'e4s_mixed', 'facedancer_mixed',
                                            'fsgan_cdf', 'faceswap_cdf', 'simswap_cdf', 'inswap_cdf', 'blendface_cdf',
                                            'uniface_cdf', 'mobileswap_cdf', 'e4s_cdf', 'facedancer_cdf', 'deepfacelab']:
                            continue
                    elif '_FR' in new_name:
                        if file_na not in ['fomm_mixed', 'facevid2vid_mixed', 'wav2lip_mixed', 'MRAA_mixed', 'one_shot_free_mixed',
                                           'fomm_cdf', 'facevid2vid_cdf', 'wav2lip_cdf', 'MRAA_cdf', 'one_shot_free_cdf',  
                                           'pirender_mixed', 'tpsm_mixed', 'lia_mixed', 'danet_mixed', 'sadtalker_mixed', 'mcnet_mixed', 'hyperreenact_mixed',
                                           'pirender_cdf', 'tpsm_cdf', 'lia_cdf', 'danet_cdf', 'sadtalker_cdf', 'mcnet_cdf', 'hyperreenact_cdf', 'heygen']:
                            continue
                    elif '_EFS' in new_name:
                        if file_na not in ['VQGAN_mixed', 'StyleGAN2_mixed', 'StyleGAN3_mixed', 'StyleGANXL_mixed', 'sd2.1_mixed',
                                            'VQGAN_cdf', 'StyleGAN2_cdf', 'StyleGAN3_cdf', 'StyleGANXL_cdf', 'sd2.1_cdf',
                                            'ddim_mixed', 'rddm_mixed', 'pixart_mixed', 'DiT_mixed', 'SiT_mixed' ,
                                            'ddim_cdf', 'rddm_cdf', 'pixart_cdf', 'DiT_cdf', 'SiT_cdf', 'MidJourney', 'whichisreal']:
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