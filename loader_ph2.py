from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from skimage import color
from color_utils import rgb_to_cielab, dataset_normalized_cielab

def dataset_normalized_cielab_loader(imgs):
    """
    Normalize CIELAB dataset for the loader
    """
    # Convert RGB to CIELAB if needed
    if imgs.max() > 100:  # Likely RGB values [0, 255]
        imgs_lab = np.zeros_like(imgs, dtype=np.float32)
        for i in range(imgs.shape[0]):
            # Convert RGB to CIELAB
            rgb_img = imgs[i].astype(np.float32) / 255.0  # Normalize to [0, 1]
            imgs_lab[i] = color.rgb2lab(rgb_img)
        imgs = imgs_lab
    
    # Now normalize the CIELAB images
    imgs_normalized = np.zeros_like(imgs, dtype=np.float32)
    
    for i in range(imgs.shape[0]):
        img_lab = imgs[i].astype(np.float32)
        
        # Per-image normalization for each channel
        for c in range(3):
            channel = img_lab[:, :, c]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            
            if std_val > 1e-7:
                channel_normalized = (channel - mean_val) / std_val
            else:
                channel_normalized = np.zeros_like(channel)
            
            # Scale to [0, 255] range for consistency
            min_val = np.min(channel_normalized)
            max_val = np.max(channel_normalized)
            if max_val - min_val > 1e-7:
                channel_normalized = ((channel_normalized - min_val) / (max_val - min_val)) * 255
            else:
                channel_normalized = np.zeros_like(channel_normalized)
            
            imgs_normalized[i, :, :, c] = channel_normalized
    
    return imgs_normalized

class isic_loader_ph2(Dataset):
    """ 
    Dataset class for PH2 CIELAB color space
    """
    def __init__(self, path_Data, train=True, Test=False):
        super(isic_loader_ph2, self).__init__()
        self.train = train
        
        if train:
            self.data = np.load(path_Data + 'data_train_ph2_cielab.npy')
            self.mask = np.load(path_Data + 'mask_train_ph2.npy')
        else:
            if Test:
                self.data = np.load(path_Data + 'data_test_ph2_cielab.npy')
                self.mask = np.load(path_Data + 'mask_test_ph2.npy')
            else:
                self.data = np.load(path_Data + 'data_val_ph2_cielab.npy')
                self.mask = np.load(path_Data + 'mask_val_ph2.npy')
        
        # Data is already in CIELAB from preparation script
        self.data = dataset_normalized_cielab_loader(self.data)
        
        # Process masks
        self.mask = np.expand_dims(self.mask, axis=3)
        if self.mask.max() > 1.0:
            self.mask = self.mask / 255.0

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        
        if self.train:
            img, seg = self.apply_augmentation(img, seg)
        
        seg = torch.tensor(seg.copy(), dtype=torch.float32)
        img = torch.tensor(img.copy(), dtype=torch.float32)
        
        img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        seg = seg.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return {'image': img, 'mask': seg}
               
    def apply_augmentation(self, img, seg):
        """Apply augmentations"""
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            seg = np.flip(seg, axis=1)
        return img, seg

    def __len__(self):
        return len(self.data)