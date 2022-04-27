#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example main file
"""

import os
from random import random
import sys
import collections
from typing import List, Dict, Tuple
import json
import csv
import pickle
from munch import Munch

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

import argparse

from ritnet.utils.utils import show_img, preprocess_image, preprocess_label, show_imgs
from ritnet.utils.config import get_config_from_json
from ritnet.model.model_builder import build_unet_model
from ritnet.dataloader.dataloader import train_generator, test_generator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def main():
  
  # Argument parsing
  MODEL_ARCHITECTURE_FOLDER = "../models"
  
  config_path = "../configs/general_config.json"
  config = get_config_from_json(config_path)

  model_config_path = "../configs/simplenet.json"
  model_config = get_config_from_json(model_config_path)

  # Create temporary variable file

  # Data loader
  train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
      tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32),
      tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.n_class), dtype=tf.float32)
    ),
    args=[config.image_size.height, config.image_size.width]
  )
  train_batch_dataset = train_dataset.batch(config.batch_size)
  train_batch_iter = iter(train_batch_dataset)

  test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
      tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32),
      tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.n_class), dtype=tf.float32)
    ),
    args=[config.image_size.height, config.image_size.width]
  )
  test_batch_dataset = test_dataset.batch(config.batch_size)
  test_batch_iter = iter(test_batch_dataset)



  # Delete temporary variable file

  pass


if __name__ == '__main__':
  main()
