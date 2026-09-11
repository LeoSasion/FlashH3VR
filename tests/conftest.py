"""Public CPU suite: no automatic downloads or GPU/model execution."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'

import torch

torch.set_num_threads(4)
