#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example dataloader
"""

import os
from typing import List, Dict, Tuple
from munch import Munch

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.python.keras.layers as layers
import numpy as np
from PIL import Image


from ..utils.utils import show_img, preprocess_image, preprocess_label, show_imgs
from ..utils.config import get_config_from_json
from ..model.model_builder import build_unet_model
from ..utils.config import GLOBAL_CONFIG

# https://www.tensorflow.org/guide/data_performance
def train_generator():

  assert GLOBAL_CONFIG.dataset_root != None\
    and GLOBAL_CONFIG.image_size.height != None\
    and GLOBAL_CONFIG.image_size.width != None, "Global Config has not been set."

  dataset_root = GLOBAL_CONFIG.dataset_root
  train_folder = [str(i) for i in range(1, 18, 1)]
  
  folders = train_folder # The list of predefined training folders [1...17]
  file_pointer = 0
  folder_pointer = 0

  while True:
    
    # Define path
    image_path = os.path.join(dataset_root, folders[folder_pointer], "synthetic")
    label_path = os.path.join(dataset_root, folders[folder_pointer], "mask-withskin")

    # Get all files from paths
    files_image = os.listdir(image_path)
    files_label = os.listdir(label_path)

    # Get image from path
    image = np.asarray(Image.open(os.path.join(image_path, files_image[file_pointer])))
    label = np.asarray(Image.open(os.path.join(label_path, files_image[file_pointer])))
    
    prep_image = preprocess_image(image, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
    prep_label = preprocess_label(label, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

    # Updating pointer to the next file 
    file_pointer += 1

    if file_pointer >= len(files_image) or file_pointer >= len(files_label):
      folder_pointer += 1
      file_pointer = 0

    if folder_pointer >= len(folders):
      folder_pointer = 0

    # Yield
    yield prep_image, prep_label

def test_generator():

  assert GLOBAL_CONFIG.dataset_root != None\
    and GLOBAL_CONFIG.image_size.height != None\
    and GLOBAL_CONFIG.image_size.width != None, "Global Config has not been set."

  dataset_root = GLOBAL_CONFIG.dataset_root
  test_folder = [str(i) for i in range(18, 25, 1)]

  folders = test_folder # The list of predefined training folders [1...17]
  file_pointer = 0
  folder_pointer = 0

  while True:
    
    # Define path
    image_path = os.path.join(dataset_root, folders[folder_pointer], "synthetic")
    label_path = os.path.join(dataset_root, folders[folder_pointer], "mask-withskin")

    # Get all files from paths
    files_image = os.listdir(image_path)
    files_label = os.listdir(label_path)

    # Get image from path
    image = np.asarray(Image.open(os.path.join(image_path, files_image[file_pointer])))
    label = np.asarray(Image.open(os.path.join(label_path, files_image[file_pointer])))
    
    prep_image = preprocess_image(image, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
    prep_label = preprocess_label(label, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

    # Updating pointer to the next file 
    file_pointer += 1

    if file_pointer >= len(files_image) or file_pointer >= len(files_label):
      folder_pointer += 1
      file_pointer = 0

    if folder_pointer >= len(folders):
      folder_pointer = 0

    # Yield
    yield prep_image, prep_label