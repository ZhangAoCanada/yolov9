import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
# from utils.callbacks import Callbacks
# from utils.dataloaders import create_dataloader
# from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
#                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
#                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
# from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
# from utils.plots import output_to_target, plot_images, plot_val_study
# from utils.torch_utils import select_device, smart_inference_mode


