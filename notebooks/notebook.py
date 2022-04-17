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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.python.keras.layers as layers
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import cv2
import tqdm

from ritnet.utils.utils import show_img, preprocess_image
from ritnet.utils.config import get_config_from_json

MODEL_ARCHITECTURE_FOLDER = "../src/model/"
dataset_root = "../data/s-general/s-general"

train_folder = [str(i) for i in range(1, 18, 1)]
test_folder = [str(i) for i in range(18, 25, 1)]

config, _ = get_config_from_json("../configs/general_config.json")

# %%

# https://www.tensorflow.org/guide/data_performance
def train_generator():

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
    
    prep_image = preprocess_image(image)
    prep_label = preprocess_image(label)

    # Updating pointer to the next file 
    file_pointer += 1

    if file_pointer >= len(files_image) or file_pointer >= len(files_label):
      folder_pointer += 1
      file_pointer = 0

    if folder_pointer >= len(folders):
      folder_pointer = 0

    # Yield
    yield prep_image, prep_label

# %%

train_dataset = tf.data.Dataset.from_generator(
  train_generator,
  output_signature=(
    tf.TensorSpec(shape=(config.image_size["height"], config.image_size["width"], config.channel), dtype=tf.float32),
    tf.TensorSpec(shape=(config.image_size["height"], config.image_size["width"], config.channel), dtype=tf.float32)
  )
)
train_batch_dataset = train_dataset.batch(config.batch_size)
train_batch_iter = iter(train_batch_dataset)


# %%

example_image, example_label = next(train_batch_iter)

# %%
example_id = 0

show_img(example_image[example_id])
show_img(example_label[example_id])



# %%

# Debugging

# example_label_path = "../data/s-general/s-general/23/mask-withskin/0000.tif"
# example_label = np.asarray(Image.open(example_label_path))
# example_label.shape

# example_image_path = "../data/s-general/s-general/23/synthetic/0000.tif"
# example_image = np.asarray(Image.open(example_image_path))

# multiply = preprocess_image(example_image) * preprocess_image(example_label)
# show_img(multiply)

