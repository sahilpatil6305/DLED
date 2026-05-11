import cv2
import numpy as np
import os
import copy

def grey_transfer_one_img(file_path, save_filter=False, save_dir=None):
    # load image
    # image_path = '/egr/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # read in grey

    # median blur
    median_blurred = cv2.medianBlur(image, 5)

    # high pass filtering
    high_pass_filtered = cv2.subtract(image, median_blurred)

    # FFT
    f_transform = np.fft.fft2(high_pass_filtered)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)

    # update save path
    save_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_grey/')
    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)

    os.makedirs(save_dir, exist_ok=True)
    filtered_image_path = os.path.join(save_dir, name+'_filtered.'+extension)
    spectrum_image_path = os.path.join(save_dir, name+'_spectrum.'+extension)

    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # save frequency map
    cv2.imwrite(spectrum_image_path, magnitude_spectrum_normalized.astype(np.uint8))

    # save high pass img
    if save_filter:
        cv2.imwrite(filtered_image_path, high_pass_filtered)

def direct_transfer_one_img(file_path, save_dir=None):
    # image_path = '/egr/DF40/DF40/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # read in grey

    # FFT
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)

    # update save path
    if save_dir == None:
        save_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_grey_d/')
    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)

    os.makedirs(save_dir, exist_ok=True)
    spectrum_image_path = os.path.join(save_dir, name+'_spectrum_d.'+extension)

    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # save frequency map
    cv2.imwrite(spectrum_image_path, magnitude_spectrum_normalized.astype(np.uint8))

def rgb_transfer_one_img_old(file_path, save_filter=False, each_channel=False, save_dir=None, version=1):
    # load img
    # image_path = '/egr/DF40/DF40/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    image = cv2.imread(file_path)  # read rgb

    # saving path
    if save_dir == None:
        save_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_rgb/')
    else:
        save_dir = save_dir+"rgb/"

    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)
    os.makedirs(save_dir, exist_ok=True)

    if save_filter:
        if save_dir == None:
            filter_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_rgb_filter/')
        else:
            filter_dir = save_dir.replace("rgb/", "filter/")
        os.makedirs(filter_dir, exist_ok=True)

    spectrum_images = []
    fileter_images = []

    # deal with each channel
    for i, color in enumerate(['B', 'G', 'R']):
        # extract one channel
        channel = image[:, :, i]
        
        # apply median blur
        median_blurred = cv2.medianBlur(channel, 5)
        
        # high pass filtering
        high_pass_filtered = cv2.subtract(channel, median_blurred)
        fileter_images.append(high_pass_filtered)
        
        # FFT
        f_transform = np.fft.fft2(high_pass_filtered)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)  # Add 1e-3 to avoid domain problems with logarithms

        # Adjust the amplitude spectrum to fit saving as an image
        if version == 1:
            magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum_normalized)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 2:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 3:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum[magnitude_spectrum<-1] = -1
            magnitude_spectrum[magnitude_spectrum>1] = 1
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)
        
        # Path to save the file
        filtered_image_path = os.path.join(filter_dir, name+f'_{color}_filtered.'+extension)
        spectrum_image_path = os.path.join(save_dir, name+f'_{color}_spectrum.'+extension)
        
        if each_channel:
            cv2.imwrite(filtered_image_path, high_pass_filtered)
            cv2.imwrite(spectrum_image_path, magnitude_spectrum_normalized.astype(np.uint8))

    # Synthetic RGB spectrogram
    combined_spectrum = cv2.merge(spectrum_images)
    combined_spectrum_rgb = cv2.cvtColor(combined_spectrum, cv2.COLOR_BGR2RGB)

    combined_spectrum_path = os.path.join(save_dir, name+'_combined_spectrum.'+extension)
    cv2.imwrite(combined_spectrum_path, combined_spectrum_rgb)

    if save_filter:
        combined_filtered = cv2.merge(fileter_images)
        combined_filtered_rgb = cv2.cvtColor(combined_filtered, cv2.COLOR_BGR2RGB)
        combined_filtered_path = os.path.join(filter_dir, name+'_combined_filtered.'+extension)
        cv2.imwrite(combined_filtered_path, combined_filtered_rgb)
        return combined_spectrum_rgb, combined_filtered_rgb
    
    return combined_spectrum_rgb

def rgb_direct_transfer_one_img(file_path, each_channel=False, save_dir=None, version=1):
    # load img
    # image_path = '/egr/research-actionlab/DF40/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    image = cv2.imread(file_path)  # read rgb

    # saving path
    if save_dir == None:
        save_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_rgb_d/')
    else:
        save_dir = save_dir+"direct/"
    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)
    os.makedirs(save_dir, exist_ok=True)

    spectrum_images = []

    # deal with each channel
    for i, color in enumerate(['B', 'G', 'R']):
        # extract one channel
        channel = image[:, :, i]
        
        # FFT
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)  # Add 1e-3 to avoid domain problems with logarithms

        # Adjust the amplitude spectrum to fit saving as an image
        if version == 1:
            magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum_normalized)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 2:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 3:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum[magnitude_spectrum<-1] = -1
            magnitude_spectrum[magnitude_spectrum>1] = 1
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)
        
        # Path to save the file
        spectrum_image_path = os.path.join(save_dir, name+f'_{color}_spectrum_d.'+extension)
        
        if each_channel:
            cv2.imwrite(spectrum_image_path, magnitude_spectrum_normalized.astype(np.uint8))

    # Synthetic RGB spectrogram
    combined_spectrum = cv2.merge(spectrum_images)
    combined_spectrum_rgb = cv2.cvtColor(combined_spectrum, cv2.COLOR_BGR2RGB)
    combined_spectrum_path = os.path.join(save_dir, name+'_combined_spectrum_d.'+extension)
    cv2.imwrite(combined_spectrum_path, combined_spectrum_rgb)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, combined_spectrum_rgb

def rgb_transfer_one_img(file_path, each_channel=False, save_dir=None, version=1, r=5):
    # load img
    # image_path = '/egr/DF40/DF40/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    image = cv2.imread(file_path)  # read rgb

    # saving path
    if save_dir == None:
        save_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_rgb/')
        filter_dir = os.path.dirname(copy.deepcopy(file_path)).replace('./datasets/DF40/', 
                                                './datasets/DF40_rgb_filter/')
    else:
        save_dir = save_dir+"rgb/"
        filter_dir = save_dir.replace("rgb/", "filter/")

    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)

    spectrum_images = []
    fileter_images = []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    rows, cols, channels = image_rgb.shape
    crow, ccol = rows // 2, cols // 2

    # Create the high pass filter mask (same mask for all channels)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    # deal with each channel
    for i, color in enumerate(['R', 'G', 'B']):
        # extract one channel
        channel = image[:, :, i]
        
        # FFT
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)

        # Apply the high pass filter mask to the FFT shifted channel
        fshift_filtered = f_shifted * mask
        
        # Compute the inverse FFT to obtain the filtered channel
        f_ishift = np.fft.ifftshift(fshift_filtered)
        channel_filtered = np.fft.ifft2(f_ishift)
        channel_filtered = np.abs(channel_filtered)
        fileter_images.append(channel_filtered)
        fshift_filtered = np.abs(fshift_filtered)
        # fshift_filtered = np.abs(f_shifted)
        spectrum_images.append(fshift_filtered)
        
        # Path to save the file
        filtered_image_path = os.path.join(filter_dir, name+f'_{color}_filtered.'+extension)
        spectrum_image_path = os.path.join(save_dir, name+f'_{color}_spectrum.'+extension)
        
        if each_channel:
            cv2.imwrite(filtered_image_path, high_pass_filtered)
            cv2.imwrite(spectrum_image_path, magnitude_spectrum_normalized.astype(np.uint8))

    # Stack the filtered channels back together to form an RGB image
    img_back = np.stack(fileter_images, axis=2)
    frequency_map = np.stack(spectrum_images, axis=2)
    
    # Normalize the result to the range [0, 255] and convert to uint8
    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    combined_filtered_rgb = img_back_norm.astype(np.uint8)

    freq_map_norm = cv2.normalize(frequency_map, None, 0, 255, cv2.NORM_MINMAX)
    combined_spectrum_rgb = freq_map_norm.astype(np.uint8)

    # save image
    combined_spectrum_path = os.path.join(save_dir, name+'_combined_spectrum.'+extension)
    cv2.imwrite(combined_spectrum_path, combined_spectrum_rgb)

    combined_filtered_path = os.path.join(filter_dir, name+'_combined_filtered.'+extension)
    cv2.imwrite(combined_filtered_path, combined_filtered_rgb)
    return image_rgb, mask, combined_spectrum_rgb, combined_filtered_rgb

def rgb_direct_transfer_map(file_path, each_channel=False, version=1):
    # load img
    # image_path = '/egr/research-actionlab/DF40/ddim/cdf/Fake_from_Youtube-real/00000/Celeb-real_id1_0008_200.png'
    img = cv2.imread(file_path)  # read rgb
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    # saving path
    img_name = file_path.rsplit('/', 1)[1]
    name, extension = img_name.rsplit('.', 1)

    spectrum_images = []

    # deal with each channel
    for i, color in enumerate(['R', 'G', 'B']):
        # extract one channel
        channel = image[:, :, i]
        
        # FFT
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)  # Add 1e-3 to avoid domain problems with logarithms

        # Adjust the amplitude spectrum to fit saving as an image
        if version == 1:
            magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum_normalized)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 2:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)
        elif version == 3:
            fft_min = np.percentile(magnitude_spectrum,5)
            fft_max = np.percentile(magnitude_spectrum,95)
            magnitude_spectrum = (magnitude_spectrum - fft_min)/(fft_max - fft_min)
            magnitude_spectrum = (magnitude_spectrum-0.5)*2
            magnitude_spectrum[magnitude_spectrum<-1] = -1
            magnitude_spectrum[magnitude_spectrum>1] = 1
            magnitude_spectrum_normalized = np.uint8(magnitude_spectrum)
            spectrum_images.append(magnitude_spectrum_normalized)

    # Synthetic RGB spectrogram
    combined_spectrum = cv2.merge(spectrum_images)
    return image, combined_spectrum



img_path = './datasets/DF40/DF40/danet/cdf/frames/id0_id1_0000/045.png'
target = './datasets/demo/DF40'
# grey_transfer_one_img(img_path)
# direct_transfer_one_img(img_path)
rgb_transfer_one_img(img_path, save_dir=target)
# rgb_direct_transfer_one_img(img_path, save_dir=target)