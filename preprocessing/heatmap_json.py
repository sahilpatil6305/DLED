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
import sys
from PIL import Image
from pathlib import Path
from frequency_map import *
from skimage.transform import resize

import torch
from torchvision import transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clip import clip

def load_clip_to_cpu(path):
    backbone_name = path
    url = clip._MODELS[backbone_name]
    root_directory = "./pretrained"
    model_path = clip._download(url, root=root_directory)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())

    return model

def normalize(img):
    """
    Normalize an image.
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = T.Normalize(mean=mean, std=std)
    return normalize(img)

def to_tensor(img):
    """
    Convert an image to a PyTorch tensor.
    """
    return T.ToTensor()(img)

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
dest_path = './datasets/heatmap/'
json_list = os.listdir(folder_path)

max_length = 2000
save_each_img = False
category_feature_dict_ff = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_feature_dict_cdf = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_feature_dict_novel = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_feature_dict = {'ff': category_feature_dict_ff, 'cdf': category_feature_dict_cdf, 'novel': category_feature_dict_novel}

category_spectrum_dict_ff = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_spectrum_dict_cdf = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_spectrum_dict_novel = { 'FS':{}, 'FR':{}, 'EFS':{}, 'FE':{}}
category_spectrum_dict = {'ff': category_spectrum_dict_ff, 'cdf': category_spectrum_dict_cdf, 'novel': category_spectrum_dict_novel}

dest_path = dest_path + str(max_length) + "/"
os.makedirs(dest_path, exist_ok=True)
print("retrieved img number for each category: ", max_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = load_clip_to_cpu("ViT-B/16")
clip_model.to(device)

with torch.no_grad():
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
                data_domain = 'cdf'
            elif '_ff' in file_na:
                data_domain = 'ff'
            else:
                assert file_na in ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                                'CollabDiff',  'stargan', 'starganv2', 'styleclip', ]
                data_domain = 'novel'

            sub_set = data[file_na]
            sub_set_keys = sub_set.keys()
            # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'

            for set_key in sub_set_keys:
                if "_Real" in set_key:
                    continue
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
                count = len(retrieved_img_list)
                spectrum_sum = 0
                image_sum = 0

                for img_path in retrieved_img_list:
                    original_image, spectrum_images_direct = rgb_direct_transfer_map(img_path)
                    rgb_img = Image.fromarray(np.array(original_image, dtype=np.uint8))
                    fre_map = Image.fromarray(np.array(spectrum_images_direct, dtype=np.uint8))

                    image = normalize(to_tensor(np.array(rgb_img))).to(device) 
                    freq_map = normalize(to_tensor(np.array(fre_map))).to(device)
                    image_features = clip_model.encode_image(image.unsqueeze(0)).cpu().numpy()
                    freq_features = clip_model.encode_image(freq_map.unsqueeze(0)).cpu().numpy()

                    image_sum += image_features
                    spectrum_sum += freq_features
                image_proto = image_sum / count
                freq_proto = spectrum_sum / count

                forgery_type = forgery_hash_dict[file_na]
                forgery_name = file_na.split('_')[0]
                category_feature_dict[data_domain][forgery_type][forgery_name] = {}
                category_feature_dict[data_domain][forgery_type][forgery_name]["proto"] = image_proto.tolist()
                category_feature_dict[data_domain][forgery_type][forgery_name]["num"] = count

                category_spectrum_dict[data_domain][forgery_type][forgery_name] = {}
                category_spectrum_dict[data_domain][forgery_type][forgery_name]["proto"] = freq_proto.tolist()
                category_spectrum_dict[data_domain][forgery_type][forgery_name]["num"] = count
                
            # else:

    # with open(dest_path+"spatial", 'w') as f:
    #     json.dump(category_feature_dict, f, indent=4)
    # with open(dest_path+"frequency", 'w') as f:
    #     json.dump(category_spectrum_dict, f, indent=4)    
    np.save(dest_path+"spatial.npy", category_feature_dict)
    np.save(dest_path+"frequency.npy", category_spectrum_dict)   
