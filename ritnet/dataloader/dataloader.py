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
import sys
from typing import List, Dict, Tuple
from munch import Munch
from scipy.ndimage import distance_transform_edt

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

def encode_label_to_distance_matrix(label: np.ndarray) -> tf.Tensor:
  """encode_label_to_distance_matrix

  https://github.com/LIVIAETS/boundary-loss/blob/8f4457416a583e33cae71443779591173e27ec62/utils.py#L260
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
  https://arxiv.org/abs/1812.07032

  Args:
    label (np.ndarray): One-hot encoded label with muultiple channels. Expect label shape to be
      (height, width, n_class). Label are expected to contain only zeros and ones.

  Returns:
    tf.Tensor: The distance matrix of the label as in the paper above.
  """
  assert len(label.shape) == 3
  
  # For every channel in the label, encode it to the distance matrix
  n_channel = label.shape[-1]

  res = []

  for channel in range(n_channel):
    positive_mask = tf.cast(label[..., channel], tf.bool) # (height, width)
    negative_mask = ~positive_mask # (height, width)
    computation_cond = tf.reduce_any(positive_mask, axis=[0, 1]) # (0,)
    computation_cond = computation_cond[..., tf.newaxis, tf.newaxis] # (1, 1)
    computation_cond = tf.broadcast_to(computation_cond, (GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width)) # (height, width)
    computation_cond = tf.cast(computation_cond, tf.double) # (height, width)

    outside_distance = distance_transform_edt(negative_mask, sampling=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
    inside_distance = distance_transform_edt(positive_mask, sampling=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

    channel_distance_matrix = outside_distance * tf.cast(negative_mask, tf.double) - (inside_distance - 1) * tf.cast(positive_mask, tf.double) # (height, width)
    channel_distance_matrix = tf.convert_to_tensor(channel_distance_matrix) # (height, width)

    channel_distance_matrix = channel_distance_matrix * computation_cond # (height, width)
    channel_distance_matrix = GLOBAL_CONFIG.sl_dist_scale * channel_distance_matrix
    channel_distance_matrix = channel_distance_matrix[..., tf.newaxis] # (height, width, 1)
    res.append(channel_distance_matrix)

  return tf.concat(res, axis=-1) # (height, width, n_channel)

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
    
    # Workaround for inconsistent data. Some input data has 3 channels,
    # while some of them has 4 channels with an empty channels in the end.
    # In that case we just remove the last channel.
    try:
      assert image.shape[-1] == 3
    except:
      if image.shape[-1] == 4:
        image = image[..., :-1]
      else:
        print("Unknown number of channel.")
        sys.exit(1)
    # Do the same check for labels.
    assert label.shape[-1] == 3, "Inconsistent label."

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
    if "sl" in GLOBAL_CONFIG.loss.name:
      dist_matrix = encode_label_to_distance_matrix(prep_label)
      yield prep_image, prep_label, dist_matrix
    else:
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
    
    # Workaround for inconsistent data. Some input data has 3 channels,
    # while some of them has 4 channels with an empty channels in the end.
    # In that case we just remove the last channel.
    try:
      assert image.shape[-1] == 3
    except:
      if image.shape[-1] == 4:
        image = image[..., :-1]
      else:
        print("Unknown number of channel.")
        sys.exit(1)
    # Do the same check for labels.
    assert label.shape[-1] == 3, "Inconsistent label."

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
    if "sl" in GLOBAL_CONFIG.loss.name:
      dist_matrix = encode_label_to_distance_matrix(prep_label)
      yield prep_image, prep_label, dist_matrix
    else:
      yield prep_image, prep_label