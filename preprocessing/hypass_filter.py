import cv2
import numpy as np
import matplotlib.pyplot as plt

def high_pass_filter_rgb(image_path, r=30):
    """
    Apply an FFT-based high pass filter to the input RGB image and then perform an inverse FFT
    to obtain the filtered image for each color channel.

    Parameters:
      image_path: Path to the input image.
      r: Radius for the high pass filter (defines the size of the central low-frequency region to be zeroed out).

    Returns:
      original: The original RGB image.
      mask: The high pass filter mask (applied to each channel, same mask for all channels).
      img_back: The RGB image after applying the high pass filter and inverse FFT.
    """
    # Read the image in color (BGR)
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        print("Cannot read image:", image_path)
        return None, None, None

    # Convert BGR to RGB
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    rows, cols, channels = original.shape
    crow, ccol = rows // 2, cols // 2

    # Create the high pass filter mask (same mask for all channels)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    # Process each channel separately
    filtered_channels = []
    for i in range(channels):
        # Extract the channel
        channel = original[:, :, i]
        
        # Compute the 2D FFT of the channel
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        
        # Apply the high pass filter mask to the FFT shifted channel
        fshift_filtered = fshift * mask
        
        # Compute the inverse FFT to obtain the filtered channel
        f_ishift = np.fft.ifftshift(fshift_filtered)
        channel_filtered = np.fft.ifft2(f_ishift)
        channel_filtered = np.abs(channel_filtered)
        
        filtered_channels.append(channel_filtered)

    # Stack the filtered channels back together to form an RGB image
    img_back = np.stack(filtered_channels, axis=2)
    
    # Normalize the result to the range [0, 255] and convert to uint8
    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back_norm = img_back_norm.astype(np.uint8)

    return original, mask, img_back_norm

if __name__ == '__main__':
    # Replace 'input.jpg' with your input image path
    image_path = 'ADE_train_00000117.jpg'
    # Adjust the radius (r) to control the extent of low-frequency removal
    original, mask, output = high_pass_filter_rgb(image_path, r=30)
    
    if original is None:
        exit(1)
    
    # Display the original image, the filter mask, and the filtered image
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original RGB Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('High Pass Filter Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(output)
    plt.title('Filtered RGB Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()