from PIL import Image
import numpy as np

fp = "/home/ad166/new/AttSwinUNet/ISIC2018/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg"  
fp1 = "/home/ad166/new/AttSwinUNet/ISIC2018/ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png"
img = Image.open(fp1)
mask = np.array(img)

print("shape :", mask.shape)
print("unique values :", np.unique(mask))