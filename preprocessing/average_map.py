import cv2
import numpy as np
import os
import glob
import argparse
import re
import json
import yaml
import pandas as pd
import random
from pathlib import Path
from frequency_map import *
from skimage.transform import resize

random.seed(123456)
category_label = ['Real', 'FS', 'FR', 'EFS', 'FE']
forgery_hash_dict={
    # train
    'VQGAN_ff':'EFS', 'StyleGAN2_ff':'EFS', 'StyleGAN3_ff':'EFS', 'StyleGANXL_ff':'EFS', 'sd2.1_ff':'EFS',  #seen
    'ddim_ff':'EFS', 'rddm_ff':'EFS', 'pixart_ff':'EFS', 'DiT_ff':'EFS', 'SiT_ff' :'EFS',                   #unseen

    'fomm_ff':'FR', 'facevid2vid_ff':'FR', 'wav2lip_ff':'FR', 'MRAA_ff':'FR', 'one_shot_free_ff':'FR',                                          #seen
    'pirender_ff':'FR', 'tpsm_ff':'FR', 'lia_ff':'FR', 'danet_ff':'FR', 'sadtalker_ff':'FR', 'mcnet_ff':'FR', 'hyperreenact_ff':'FR',           #unseen

    'fsgan_ff':'FS', 'faceswap_ff':'FS', 'simswap_ff':'FS', 'inswap_ff':'FS', 'blendface_ff':'FS',                          #seen
    'uniface_ff':'FS', 'mobileswap_ff':'FS', 'e4s_ff':'FS', 'facedancer_ff':'FS',                                            #unseen

    # test
    "e4e_ff":'FE', "e4e_cdf":'FE', 'CollabDiff':'FE', 'stargan':'FE', 'starganv2':'FE', 'styleclip':'FE',                   #unseen

    'VQGAN_cdf':'EFS', 'StyleGAN2_cdf':'EFS', 'StyleGAN3_cdf':'EFS', 'StyleGANXL_cdf':'EFS', 'sd2.1_cdf':'EFS',                             #seen
    'ddim_cdf':'EFS', 'rddm_cdf':'EFS', 'pixart_cdf':'EFS', 'DiT_cdf':'EFS', 'SiT_cdf':'EFS', 'MidJourney':'EFS', 'whichisreal':'EFS',      #unseen

    'fomm_cdf':'FR', 'facevid2vid_cdf':'FR', 'wav2lip_cdf':'FR', 'MRAA_cdf':'FR', 'one_shot_free_cdf':'FR',                                                   #seen
    'pirender_cdf':'FR', 'tpsm_cdf':'FR', 'lia_cdf':'FR', 'danet_cdf':'FR', 'sadtalker_cdf':'FR', 'mcnet_cdf':'FR', 'hyperreenact_cdf':'FR', 'heygen':'FR',   #unseen

    'fsgan_cdf':'FS', 'faceswap_cdf':'FS', 'simswap_cdf':'FS', 'inswap_cdf':'FS', 'blendface_cdf':'FS',                 #seen
    'uniface_cdf':'FS', 'mobileswap_cdf':'FS', 'e4s_cdf':'FS', 'facedancer_cdf':'FS', 'deepfacelab':'FS'                #unseen
}

folder_path = './datasets/DF40/json_used/'
dest_path = './datasets/illustration/'
json_list = os.listdir(folder_path)

max_length = 20
save_each_img = False
category_spectrum_dict_ff = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_filter_dict_ff = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_direct_dict_ff = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}

category_spectrum_dict_cdf = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_filter_dict_cdf = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_direct_dict_cdf = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}

category_spectrum_dict_novel = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_filter_dict_novel = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}
category_direct_dict_novel = {'Real':None, 'FS':None, 'FR':None, 'EFS':None, 'FE':None}

category_spectrum_dict = {'ff': category_spectrum_dict_ff, 'cdf': category_spectrum_dict_cdf, 'novel': category_spectrum_dict_novel}
category_filter_dict = {'ff': category_filter_dict_ff, 'cdf': category_filter_dict_cdf, 'novel': category_filter_dict_novel}
category_direct_dict = {'ff': category_direct_dict_ff, 'cdf': category_direct_dict_cdf, 'novel': category_direct_dict_novel}

dest_path = dest_path + str(max_length) + "/"
print("retrieved img number for each category: ", max_length)

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
        
        print("Dealing with file: ", file_na)
        if '_cdf' in file_na:
            target_path = dest_path + 'cdf/'
            data_domain = 'cdf'
        elif '_ff' in file_na:
            target_path = dest_path + 'ff/'
            data_domain = 'ff'
        else:
            assert file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                               'CollabDiff',  'stargan', 'starganv2', 'styleclip', ]
            target_path = dest_path + 'novel/'
            data_domain = 'novel'

        sub_set = data[file_na]
        sub_set_keys = sub_set.keys()
        # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'

        for set_key in sub_set_keys:
            retrieved_img_list = []
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
                    retrieved_img_list.extend(file_path_list)
                    # file_path_list:
                    #  ['./datasets/DF40/FaceForensics++/original_sequences/youtube/c23/frames/071/277.png', 
                    # './datasets/DF40/FaceForensics++/original_sequences/youtube/c23/frames/071/349.png'....]
            if len(retrieved_img_list) > max_length:
                retrieved_img_list = random.sample(retrieved_img_list, max_length)

            """
            finegrained demonstraton for each forgery, E.x: fsgan_Real vs fsgan_Fake
            """
            spectrum_sum = None
            filter_sum = None
            direct_sum = None
            count = 0

            finegrained_forgery_path = target_path+file_na+"/"+set_key+"/"
            os.makedirs(finegrained_forgery_path, exist_ok=True)
            for img_path in retrieved_img_list:
                image_rgb, mask, spectrum_images, fileter_images = rgb_transfer_one_img(img_path, save_dir=finegrained_forgery_path)
                spectrum_images_direct = rgb_direct_transfer_one_img(img_path, save_dir=finegrained_forgery_path)

                if spectrum_sum is None:
                    spectrum_sum = np.zeros_like(spectrum_images, dtype=np.float32)
                    filter_sum = np.zeros_like(fileter_images, dtype=np.float32)
                    direct_sum = np.zeros_like(spectrum_images_direct, dtype=np.float32)
                
                spectrum_sum += spectrum_images
                filter_sum += fileter_images
                direct_sum += spectrum_images_direct
                count += 1
            spectrum_mean = (spectrum_sum / count).astype(np.uint8)
            filter_mean = (filter_sum / count).astype(np.uint8)
            direct_mean = (direct_sum / count).astype(np.uint8)
            cv2.imwrite(os.path.join(finegrained_forgery_path, 'spectrum_mean.png'), spectrum_mean)
            cv2.imwrite(os.path.join(finegrained_forgery_path, 'filter_mean.png'), filter_mean)
            cv2.imwrite(os.path.join(finegrained_forgery_path, 'direct_mean.png'), direct_mean)
        
            if spectrum_sum.shape[0] != 256:
                spectrum_sum = resize(spectrum_sum, (256,256,3), preserve_range=True).astype(np.float32)
                filter_sum = resize(filter_sum, (256,256,3), preserve_range=True).astype(np.float32)
                direct_sum = resize(direct_sum, (256,256,3), preserve_range=True).astype(np.float32)
            elif spectrum_sum.shape[1] != 256:
                spectrum_sum = resize(spectrum_sum, (256,256,3), preserve_range=True).astype(np.float32)
                filter_sum = resize(filter_sum, (256,256,3), preserve_range=True).astype(np.float32)
                direct_sum = resize(direct_sum, (256,256,3), preserve_range=True).astype(np.float32)

            if "_Real" in set_key:
                if category_spectrum_dict[data_domain]['Real'] == None:
                    category_spectrum_dict[data_domain]['Real']=[spectrum_sum, count]
                    category_filter_dict[data_domain]['Real']=[filter_sum, count]
                    category_direct_dict[data_domain]['Real']=[direct_sum, count]
                else:
                    category_spectrum_dict[data_domain]['Real'][0] += spectrum_sum
                    category_spectrum_dict[data_domain]['Real'][1] += count
                    category_filter_dict[data_domain]['Real'][0] += filter_sum
                    category_filter_dict[data_domain]['Real'][1] += count
                    category_direct_dict[data_domain]['Real'][0] += direct_sum
                    category_direct_dict[data_domain]['Real'][1] += count
            else:
                forgery_type = forgery_hash_dict[file_na]
                if category_spectrum_dict[data_domain][forgery_type] == None:
                    category_spectrum_dict[data_domain][forgery_type]=[spectrum_sum, count]
                    category_filter_dict[data_domain][forgery_type]=[filter_sum, count]
                    category_direct_dict[data_domain][forgery_type]=[direct_sum, count]
                else:
                    category_spectrum_dict[data_domain][forgery_type][0] += spectrum_sum
                    category_spectrum_dict[data_domain][forgery_type][1] += count
                    category_filter_dict[data_domain][forgery_type][0] += filter_sum
                    category_filter_dict[data_domain][forgery_type][1] += count
                    category_direct_dict[data_domain][forgery_type][0] += direct_sum
                    category_direct_dict[data_domain][forgery_type][1] += count 


"""
category demonstraton
"""
for d_domain in ['ff', 'cdf', 'novel']:
    tar_path = dest_path + 'overall/' + d_domain + '/'
    for cate_name in category_label:
        if category_spectrum_dict[d_domain][cate_name] != None:
            print(d_domain, " : ", cate_name, " : ", "sample number: ", category_spectrum_dict[d_domain][cate_name][1])
            coarse_category_path = tar_path + cate_name + '/'
            os.makedirs(coarse_category_path, exist_ok=True)
            spectrum_mean = (category_spectrum_dict[d_domain][cate_name][0] / category_spectrum_dict[d_domain][cate_name][1]).astype(np.uint8)
            filter_mean = (category_filter_dict[d_domain][cate_name][0] / category_filter_dict[d_domain][cate_name][1]).astype(np.uint8)
            direct_mean = (category_direct_dict[d_domain][cate_name][0] / category_direct_dict[d_domain][cate_name][1]).astype(np.uint8)
            cv2.imwrite(os.path.join(coarse_category_path, 'spectrum_mean.png'), spectrum_mean)
            cv2.imwrite(os.path.join(coarse_category_path, 'filter_mean.png'), filter_mean)
            cv2.imwrite(os.path.join(coarse_category_path, 'direct_mean.png'), direct_mean)
        # else:
        