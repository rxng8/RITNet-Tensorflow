#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example notebook file
"""

# %%

import os
import sys
import collections
from typing import List, Dict, Tuple
import json
import csv
import pickle

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.python.keras.layers as layers
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import cv2
import tqdm

from ritnet.utils.utils import show_img, preprocess_image

MODEL_ARCHITECTURE_FOLDER = "../src/model/"
dataset_root = "../data/s-general"

train_folder = [i for i in range(1, 16, 1)]
test_folder = [i for i in range(16, 25, 1)]



# %%

def train_generator():
  pass

# %%

img = Image.open("../data/s-general/s-general/1/synthetic/0000.tif")

np_img = np.asarray(img)

np_img.shape


# %%


tf.reduce_max(np_img)
