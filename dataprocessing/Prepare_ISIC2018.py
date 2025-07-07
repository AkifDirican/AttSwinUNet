# -*- coding: utf-8 -*-
"""
Updated ISIC 2018 data preparation script for CIELAB color space
"""
import numpy as np
from PIL import Image
from skimage import color
import glob

# Parameters
height = 224
width = 224
channels = 3

def convert_rgb_to_cielab(rgb_image):
    """Convert RGB image to CIELAB"""
    # Ensure image is in [0, 1] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    
    # Convert to CIELAB
    lab_image = color.rgb2lab(rgb_image)
    return lab_image

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = '/home/ad166/new/AttSwinUNet/ISIC2018/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')

# It contains 2594 training samples
Data_train_2018 = np.zeros([2594, height, width, channels])
Label_train_2018 = np.zeros([2594, height, width])

print('Reading ISIC 2018 and converting to CIELAB')
for idx in range(len(Tr_list)):
    if idx % 100 == 0:
        print(f'Processing image {idx+1}/{len(Tr_list)}')
    
    # Load RGB image
    img = np.array(Image.open(Tr_list[idx]))
    img = np.array(Image.fromarray(img.astype('uint8')).resize((width, height), Image.BILINEAR))
    
    # Convert RGB to CIELAB
    img_rgb_normalized = img.astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb_normalized)
    
    Data_train_2018[idx, :, :, :] = img_lab

    # Load mask (unchanged)
    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a + 'ISIC2018_Task1_Training_GroundTruth/' + b + '_segmentation.png')    
    img2 = np.array(Image.open(add))
    img2 = np.array(Image.fromarray(img2.astype('uint8')).resize((width, height), Image.BILINEAR), dtype=np.double)
    Label_train_2018[idx, :, :] = img2 
         
print('Reading ISIC 2018 finished - Data is now in CIELAB color space')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img = Data_train_2018[0:1815, :, :, :]
Validation_img = Data_train_2018[1815:1815+259, :, :, :]
Test_img = Data_train_2018[1815+259:2594, :, :, :]

Train_mask = Label_train_2018[0:1815, :, :]
Validation_mask = Label_train_2018[1815:1815+259, :, :]
Test_mask = Label_train_2018[1815+259:2594, :, :]

# Save the CIELAB data
np.save('data_train_cielab', Train_img)
np.save('data_test_cielab', Test_img)
np.save('data_val_cielab', Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test', Test_mask)
np.save('mask_val', Validation_mask)

print("Data saved in CIELAB format")
print("CIELAB ranges:")
print(f"L* channel: [{np.min(Data_train_2018[:,:,:,0]):.2f}, {np.max(Data_train_2018[:,:,:,0]):.2f}]")
print(f"a* channel: [{np.min(Data_train_2018[:,:,:,1]):.2f}, {np.max(Data_train_2018[:,:,:,1]):.2f}]")
print(f"b* channel: [{np.min(Data_train_2018[:,:,:,2]):.2f}, {np.max(Data_train_2018[:,:,:,2]):.2f}]")