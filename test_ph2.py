from __future__ import division
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loader_ph2 import isic_loader_ph2  # Use PH2 loader
import glob
import numpy as np
import copy
import yaml
from tqdm import tqdm
from model.attention_swin_unet import SwinAttentionUnet as ViT_seg
from sklearn.metrics import confusion_matrix,f1_score
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
import argparse
from configs import swin_attention_unet as config
from types import SimpleNamespace 
from inference import inference

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./Synapse/', help='root dir for data')
parser.add_argument('--eval_interval', type=int, default=5, help='eval interval')
parser.add_argument('--volume_path', type=str,
                    default='./OUT_DIR/best_model_ph2_final.pth', help='the weight path')
parser.add_argument('--dataset', type=str,
                    default='PH2', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--saved_model', type=str,
                    default='./OUT_DIR/best_model_ph2_final.pth' , help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--mode', help='Select our contribution',
                    choices=['swin','cross_contextual_attention'], default='swin')
parser.add_argument('--skip_num', help='Select our contribution',
                    choices=['0', '1', '2','3'], default='3'),
parser.add_argument('--operationaddatten', help='Select our contribution',
                    choices=['+', 'mul'], default='+')
parser.add_argument('--spatial_attention', help='0 or 1',
                    choices=['0',"1"], default="0")
parser.add_argument('--isxvit', type=str, 
                    choices=['0', '1'], default='1',
                    help='Enable CrossViT layers')

device = 'cuda' if torch.cuda.is_available() else 'cpu'               
args = parser.parse_args()
config_model = config.get_swin_unet_attention_configs().to_dict()
config_model.update(vars(args))
configs = SimpleNamespace(**config_model)

# Load PH2 config
config = yaml.load(open('./configs/config_ph2.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
patience = 0
data_path = config['path_to_data']

# Use PH2 test dataset
test_dataset = isic_loader_ph2(path_Data=data_path, train=False, Test=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model
Net = ViT_seg(configs, num_classes=args.num_classes).cuda()
Net = Net.to(device)

print(f"Test config: mode={configs.mode}, spatial_attention={configs.spatial_attention}")
print(f"Loading PH2 fine-tuned model from: {configs.volume_path}")

# Load weights with error handling
try:
    checkpoint = torch.load(configs.volume_path, map_location='cpu')
    if 'model_weights' in checkpoint:
        Net.load_state_dict(checkpoint['model_weights'])
        print(f"Successfully loaded PH2 fine-tuned model")
        if 'test_F1_score' in checkpoint:
            print(f"Model was saved with F1 score: {checkpoint['test_F1_score']:.4f}")
    else:
        Net.load_state_dict(checkpoint)
    print(f"Successfully loaded model from {configs.volume_path}")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("\nMake sure you're using the same arguments as training:")
    print("  --mode cross_contextual_attention")
    print("  --spatial_attention 1")
    print("  --isxvit 1")
    exit(1)
except FileNotFoundError:
    print(f"Model file not found: {configs.volume_path}")
    print("Make sure you've completed the PH2 fine-tuning stage first.")
    exit(1)

# Run inference
print(f"\nRunning inference on PH2 test set...")
print(f"Test set size: {len(test_dataset)} samples")
inference(Net, test_loader)