import os
import json
import copy
from PIL import Image

folder_path = './datasets/DF40/data_json'
dest_path = './datasets/DF40/json_used/'
data_path = './datasets/DF40/'
# all_data = {}
json_list = os.listdir(folder_path)
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            new_data = copy.deepcopy(data)
            # all_data[filename]=data
        file_na, _ = os.path.splitext(filename)
        # e.g. file_na = "fsgan_ff" for "fsgan_ff.json"
        if file_na == 'uniface_ori':
            file_na = file_na.replace("_ori",'')
        sub_set=new_data[file_na]
        sub_set_keys = sub_set.keys()
        # e.g. sub_set keys: 'fsgan_Real', 'fsgan_Fake'
        for set_key in sub_set_keys:
            spec_img_set = sub_set[set_key]
            spec_keys = spec_img_set.keys()
            # spec keys: train; test; val
            for spec_key in spec_keys:
                de_img_set = spec_img_set[spec_key]
                de_keys = de_img_set.keys()
                # e.g. '071_fsgan', '054_fsgan'
                for attributes in de_keys:
                    attribute_set = de_img_set[attributes]
                    at_keys = attribute_set.keys()
                    # at_keys: 'label' 'frames'
                    file_path_list = attribute_set['frames']
                    del_index_list =[]
                    for im_index, image_path in enumerate(file_path_list):
                        if file_na in ['e4e_cdf','e4e_ff']:
                            new_image_path = image_path.replace('<OLD_DATASET_ROOT>','./datasets/DF40')
                            file_path_list[im_index] = new_image_path
                        elif file_na == 'MidJourney':
                            new_image_path = image_path.replace('.gif','.png')
                            new_image_path = new_image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            if new_image_path in [data_path+'DF40/MidJourney/fake/13592447708_Single_person_East_Asian_male_looking_at_the_camera_75e759ff-c23a-4679-bd68-76170370d3d1.png',
                                                  data_path+'DF40/MidJourney/fake/13592447708_Single_person_white_female_looking_at_the_camera_fa_72edc876-2d1f-41b6-b7cb-caaa907da5ab.png',
                                                  data_path+'DF40/MidJourney/fake/13592447708_Clean_East_Asian_female_face_face_only_shown_face_c_ba354e74-3cb4-4f63-9ad6-f40a641b558a.png',
                                                  data_path+'DF40/MidJourney/fake/13592447708_Clean_East_Asian_male_face_only_face_shown_close-up_3ad609fd-9d87-4242-9d5b-d2776a3c2c7e.png',
                                                  data_path+'DF40/MidJourney/fake/13592447708_Single_person_white_female_looking_at_the_camera_fa_d28fc222-bd3f-4ee6-ad9f-7af441607d82.png']:
                                del_index_list.append(im_index)
                                continue
                            file_path_list[im_index] = new_image_path
                        elif file_na == 'whichisreal':
                            new_image_path = image_path.replace('whichisreal','whichfaceisreal')
                            new_image_path = new_image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            file_path_list[im_index] = new_image_path
                        elif file_na == 'starganv2':
                            new_image_path = image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            if new_image_path in [data_path+'DF40/starganv2/fake/1096.jpg',
                                                  data_path+'DF40/starganv2/real/1132.jpg',
                                                  data_path+'DF40/starganv2/fake/779.jpg',
                                                  data_path+'DF40/starganv2/real/547.jpg']:
                                del_index_list.append(im_index)
                                continue
                            file_path_list[im_index] = new_image_path
                        elif file_na in ['rddm_ff', 'rddm_cdf']:
                            new_image_path = image_path.replace('rddm','RDDM')
                            if '_Fake' in set_key:
                                new_image_path = new_image_path.replace('/Celeb-real/','/Fake_from_Celeb-real/')
                                new_image_path = new_image_path.replace('/YouTube-real/','/Fake_from_Youtube-real/')
                            if spec_key == 'train':
                                new_image_path = new_image_path.replace('DF40','DF40_train')
                                new_image_path = new_image_path.replace('ff/','')
                            new_image_path = new_image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            file_path_list[im_index] = new_image_path
                        elif file_na in ['pixart_ff','ddim_ff','DiT_ff','sd2.1_ff','SiT_ff',
                                         'StyleGAN2_ff','StyleGAN3_ff','StyleGANXL_ff','VQGAN_ff']:
                            if spec_key == 'train':
                                new_image_path = image_path.replace('DF40','DF40_train')
                                new_image_path = new_image_path.replace('ff/','')
                                new_image_path = new_image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            else:
                                new_image_path = image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            file_path_list[im_index] = new_image_path
                        else:
                            new_image_path = image_path.replace('deepfakes_detection_datasets','./datasets/DF40')
                            if '_cdf' in file_na and file_na != 'sd2.1_cdf':
                                if '_Fake' in set_key:
                                    new_image_path = new_image_path.replace('/Celeb-real/','/Fake_from_Celeb-real/')
                                    new_image_path = new_image_path.replace('/YouTube-real/','/Fake_from_Youtube-real/')
                            file_path_list[im_index] = new_image_path
                    for dw_index, _ in enumerate(del_index_list):
                        del_pos = int(del_index_list[dw_index]-dw_index)
                        del file_path_list[del_pos]
                    # break
                # break
            # break
        
        with open(os.path.join(dest_path, filename), 'w') as fdest:
            json.dump(new_data, fdest, indent=4)
    # break


MidJourney_path = './datasets/DF40/DF40/MidJourney/fake'
img_list = os.listdir(MidJourney_path)
for filename in os.listdir(MidJourney_path):
    if filename.endswith('.gif'):
        gif_path = os.path.join(MidJourney_path, filename)
        gif_image = Image.open(gif_path)
        file_na, _ = os.path.splitext(filename)
        out_path = os.path.join(MidJourney_path, file_na+'.png')
        gif_image.save(out_path, 'PNG')