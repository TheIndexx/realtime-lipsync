import os
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs
from musetalk.utils.blending import get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model
import shutil
import queue
import time

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()
