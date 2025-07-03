import numpy as np
import cv2
from skimage import color
import torch

def rgb_to_cielab(rgb_image):
    """
    Convert RGB image to CIELAB color space
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255] or [0, 1]
    
    Returns:
        lab_image: numpy array of shape (H, W, 3) in CIELAB space
    """
    # Ensure input is in [0, 1] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    
    # Convert to CIELAB using scikit-image
    lab_image = color.rgb2lab(rgb_image)
    
    return lab_image

def cielab_to_rgb(lab_image):
    """
    Convert CIELAB image back to RGB
    
    Args:
        lab_image: numpy array of shape (H, W, 3) in CIELAB space
    
    Returns:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]
    """
    # Convert back to RGB
    rgb_image = color.lab2rgb(lab_image)
    
    # Scale to [0, 255] and convert to uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

def normalize_cielab(lab_image):
    """
    Normalize CIELAB image for neural network input
    
    CIELAB ranges:
    - L*: [0, 100]
    - a*: [-127, 127] 
    - b*: [-127, 127]
    
    Args:
        lab_image: numpy array of shape (H, W, 3) in CIELAB space
    
    Returns:
        normalized_lab: numpy array with normalized values
    """
    lab_normalized = lab_image.copy().astype(np.float32)
    
    # Normalize L* channel from [0, 100] to [0, 1]
    lab_normalized[:, :, 0] = lab_normalized[:, :, 0] / 100.0
    
    # Normalize a* and b* channels from [-127, 127] to [-1, 1]
    lab_normalized[:, :, 1] = lab_normalized[:, :, 1] / 127.0
    lab_normalized[:, :, 2] = lab_normalized[:, :, 2] / 127.0
    
    return lab_normalized

def denormalize_cielab(normalized_lab):
    """
    Denormalize CIELAB image back to standard ranges
    
    Args:
        normalized_lab: numpy array with normalized CIELAB values
    
    Returns:
        lab_image: numpy array in standard CIELAB ranges
    """
    lab_image = normalized_lab.copy().astype(np.float32)
    
    # Denormalize L* channel from [0, 1] to [0, 100]
    lab_image[:, :, 0] = lab_image[:, :, 0] * 100.0
    
    # Denormalize a* and b* channels from [-1, 1] to [-127, 127]
    lab_image[:, :, 1] = lab_image[:, :, 1] * 127.0
    lab_image[:, :, 2] = lab_image[:, :, 2] * 127.0
    
    return lab_image

def dataset_normalized_cielab(imgs):
    """
    Normalize CIELAB dataset similar to the original RGB normalization
    but adapted for CIELAB color space
    
    Args:
        imgs: numpy array of shape (N, H, W, 3) in CIELAB space
    
    Returns:
        normalized_imgs: normalized images ready for training
    """
    # Convert to CIELAB if not already
    if imgs.max() > 100:  # Likely still in RGB
        imgs_lab = np.zeros_like(imgs)
        for i in range(imgs.shape[0]):
            imgs_lab[i] = rgb_to_cielab(imgs[i])
        imgs = imgs_lab
    
    # Normalize each channel separately
    imgs_normalized = np.zeros_like(imgs, dtype=np.float32)
    
    for i in range(imgs.shape[0]):
        img_lab = imgs[i].astype(np.float32)
        
        # Normalize L* channel (0-100 range)
        l_channel = img_lab[:, :, 0]
        l_mean = np.mean(l_channel)
        l_std = np.std(l_channel)
        if l_std > 1e-7:
            l_normalized = (l_channel - l_mean) / l_std
        else:
            l_normalized = np.zeros_like(l_channel)
        
        # Normalize a* channel (-127 to 127 range)
        a_channel = img_lab[:, :, 1]
        a_mean = np.mean(a_channel)
        a_std = np.std(a_channel)
        if a_std > 1e-7:
            a_normalized = (a_channel - a_mean) / a_std
        else:
            a_normalized = np.zeros_like(a_channel)
        
        # Normalize b* channel (-127 to 127 range)
        b_channel = img_lab[:, :, 2]
        b_mean = np.mean(b_channel)
        b_std = np.std(b_channel)
        if b_std > 1e-7:
            b_normalized = (b_channel - b_mean) / b_std
        else:
            b_normalized = np.zeros_like(b_channel)
        
        # Combine normalized channels
        img_normalized = np.stack([l_normalized, a_normalized, b_normalized], axis=2)
        
        # Scale to [0, 255] range for consistency with original implementation
        min_val = np.min(img_normalized)
        max_val = np.max(img_normalized)
        if max_val - min_val > 1e-7:
            img_normalized = ((img_normalized - min_val) / (max_val - min_val)) * 255
        else:
            img_normalized = np.zeros_like(img_normalized)
        
        imgs_normalized[i] = img_normalized
    
    return imgs_normalized