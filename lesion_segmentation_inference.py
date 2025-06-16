import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from types import SimpleNamespace
from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from model.attention_swin_unet import SwinAttentionUnet as ViT_seg
from configs import swin_attention_unet as config

# Fixed paths
MODEL_PATH = "OUT_DIR/best_model.pth"
INPUT_DIR = "/home/ad166/new/AttSwinUNet/ISIC2018/ISIC2018_Task1-2_Training_Input/"
GROUND_TRUTH_DIR = "/home/ad166/new/AttSwinUNet/ISIC2018/ISIC2018_Task1_Training_GroundTruth/"
OUTPUT_DIR = "image_trials"

def load_model(model_path, configs, device):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = ViT_seg(configs, num_classes=1).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_weights' in checkpoint:
        model.load_state_dict(checkpoint['model_weights'])
        print(f"Model loaded with F1 score: {checkpoint.get('test_F1_score', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def preprocess_image(image_path, target_size=224):
    """Preprocess a single image for inference"""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    # Resize image
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize (similar to dataset_normalized function in loader.py)
    image_normalized = image_resized.astype(np.float32)
    
    # Normalize to 0-1
    image_normalized = image_normalized / 255.0
    
    # Additional normalization (similar to the loader)
    mean = np.mean(image_normalized)
    std = np.std(image_normalized)
    if std == 0:
        std = 1e-8
    image_normalized = (image_normalized - mean) / std
    
    # Scale to 0-255
    min_val = np.min(image_normalized)
    max_val = np.max(image_normalized)
    if max_val - min_val == 0:
        image_normalized = np.zeros_like(image_normalized)
    else:
        image_normalized = ((image_normalized - min_val) / (max_val - min_val)) * 255
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, image_resized

def load_ground_truth(image_number):
    """Load ground truth mask for given image number"""
    gt_filename = f"ISIC_{image_number:07d}_segmentation.png"
    gt_path = os.path.join(GROUND_TRUTH_DIR, gt_filename)
    
    if not os.path.exists(gt_path):
        raise ValueError(f"Ground truth not found: {gt_path}")
    
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    gt_normalized = gt_resized.astype(np.float32) / 255.0
    
    return gt_normalized

def postprocess_prediction(prediction, threshold=0.43, morphology_size=6):
    """Post-process the model prediction"""
    # Apply threshold
    binary_pred = np.where(prediction >= threshold, 1, 0).astype(np.uint8)
    
    # Apply morphological operations
    structure = np.ones((morphology_size, morphology_size))
    binary_pred = binary_opening(binary_pred, structure=structure).astype(np.uint8)
    binary_pred = binary_fill_holes(binary_pred, structure=structure).astype(np.uint8)
    
    return binary_pred

def visualize_results(original_image, prediction, ground_truth, image_name=""):
    """Visualize the segmentation results with 3 panels"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image: {image_name}')
    axes[0].axis('off')
    
    # Model prediction
    axes[1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Model Prediction')
    axes[1].axis('off')
    
    # Difference between ground truth and prediction
    difference = np.abs(ground_truth - prediction)
    im = axes[2].imshow(difference, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('|Ground Truth - Prediction|')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def inference_single_image(model, image_number, device, configs):
    """Run inference on a single image by number"""
    # Construct image path
    image_filename = f"ISIC_{image_number:07d}.jpg"
    image_path = os.path.join(INPUT_DIR, image_filename)
    
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    
    print(f"Processing: {image_filename}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, target_size=configs.image_size)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        if image_tensor.size(1) == 1:  # Handle grayscale
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
        
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction_np = prediction.cpu().numpy()[0, 0]
    
    # Load ground truth
    try:
        ground_truth = load_ground_truth(image_number)
    except ValueError as e:
        print(f"Warning: {e}")
        ground_truth = np.zeros_like(prediction_np)
    
    # Post-process prediction
    binary_prediction = postprocess_prediction(prediction_np)
    
    # Visualize
    fig = visualize_results(original_image, binary_prediction, ground_truth, image_filename)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = f"ISIC_{image_number:07d}"
    
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png"), 
               dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate and print metrics
    intersection = np.sum(binary_prediction * ground_truth)
    union = np.sum(binary_prediction) + np.sum(ground_truth) - intersection
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (np.sum(binary_prediction) + np.sum(ground_truth)) if (np.sum(binary_prediction) + np.sum(ground_truth)) > 0 else 0
    
    print(f"IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    return prediction_np, binary_prediction, ground_truth

def main():
    if len(sys.argv) < 2:
        print("Usage: python lesion_segmentation_inference.py <image_number1> <image_number2> ...")
        print("Example: python lesion_segmentation_inference.py 2 10")
        sys.exit(1)
    
    # Parse image numbers from command line
    image_numbers = []
    for arg in sys.argv[1:]:
        try:
            image_numbers.append(int(arg))
        except ValueError:
            print(f"Invalid image number: {arg}")
            sys.exit(1)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup configurations
    configs_dict = config.get_swin_unet_attention_configs().to_dict()
    configs_dict.update({
        'mode': 'cross_contextual_attention',
        'spatial_attention': '1',
        'isxvit': '1',
        'skip_num': '3',
        'operationaddatten': '+',
        'num_classes': 1
    })
    configs = SimpleNamespace(**configs_dict)
    
    # Load model
    model = load_model(MODEL_PATH, configs, device)
    
    # Process each image
    for image_number in image_numbers:
        try:
            inference_single_image(model, image_number, device, configs)
        except Exception as e:
            print(f"Error processing image {image_number}: {str(e)}")
            continue

if __name__ == "__main__":
    main()