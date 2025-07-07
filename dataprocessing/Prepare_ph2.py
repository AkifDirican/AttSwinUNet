# -*- coding: utf-8 -*-
"""
Updated ph2 data preparation script for CIELAB color space
"""

import numpy as np
from PIL import Image
from skimage import color
import glob
import os

# Parameters
height = 224
width  = 224
channels = 3

def convert_rgb_to_cielab(rgb_image):
    """Convert RGB image to CIELAB"""
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    lab_image = color.rgb2lab(rgb_image)
    return lab_image

############################################################# Prepare ph2 data set #################################################
Dataset_add = '/home/ad166/new/AttSwinUNet/PH2Dataset/PH2 Dataset images'

# Each image is in a subfolder named IMDxxx
image_folders = sorted(glob.glob(os.path.join(Dataset_add, 'IMD*')))

num_samples = len(image_folders)
Data_ph2 = np.zeros([num_samples, height, width, channels], dtype=np.float32)
Label_ph2 = np.zeros([num_samples, height, width], dtype=np.float32)


print('Reading PH2 and converting to CIELAB')
for idx, folder in enumerate(image_folders):
    if idx % 20 == 0:
        print(f'Processing image {idx+1}/{num_samples}')
    # Find dermoscopic image
    derm_img_path = glob.glob(os.path.join(folder, '*_Dermoscopic_Image', '*.bmp'))
    if len(derm_img_path) == 0:
        derm_img_path = glob.glob(os.path.join(folder, '*_Dermoscopic_Image', '*.png'))
    if len(derm_img_path) == 0:
        print(f"Could not find dermoscopic image in {folder}")
        continue
    img = np.array(Image.open(derm_img_path[0]))
    img = np.array(Image.fromarray(img.astype('uint8')).resize((width, height), Image.BILINEAR))
    img_lab = convert_rgb_to_cielab(img)
    Data_ph2[idx, :, :, :] = img_lab

    # Find lesion mask
    mask_path = glob.glob(os.path.join(folder, '*_lesion', '*.bmp'))
    if len(mask_path) == 0:
        mask_path = glob.glob(os.path.join(folder, '*_lesion', '*.png'))
    if len(mask_path) == 0:
        print(f"Could not find lesion mask in {folder}")
        continue
    mask = np.array(Image.open(mask_path[0]))
    # If mask is RGB, convert to grayscale
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.array(Image.fromarray(mask.astype('uint8')).resize((width, height), Image.BILINEAR), dtype=np.float32)
    Label_ph2[idx, :, :] = mask

print('Reading PH2 finished - Data is now in CIELAB color space')

################################################################ Make the train and test sets ########################################    
# Example split: 80 train, 20 val, rest test (adjust as needed)
n_train = 80
n_val = 20
n_test = num_samples - n_train - n_val

Train_img = Data_ph2[0:n_train, :, :, :]
Validation_img = Data_ph2[n_train:n_train+n_val, :, :, :]
Test_img = Data_ph2[n_train+n_val:, :, :, :]

Train_mask = Label_ph2[0:n_train, :, :]
Validation_mask = Label_ph2[n_train:n_train+n_val, :, :]
Test_mask = Label_ph2[n_train+n_val:, :, :]

# Save the CIELAB data
np.save('data_train_ph2_cielab', Train_img)
np.save('data_test_ph2_cielab', Test_img)
np.save('data_val_ph2_cielab', Validation_img)

np.save('mask_train_ph2', Train_mask)
np.save('mask_test_ph2', Test_mask)
np.save('mask_val_ph2', Validation_mask)