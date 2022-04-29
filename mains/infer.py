#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example main file
"""
# %%

import os
from random import random
import sys
import collections
from typing import List, Dict, Tuple
import json
import csv
import pickle
from munch import Munch
import random

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

from ritnet.trainer.trainer import Trainer
from ritnet.utils.utils import show_img, preprocess_image, preprocess_label, show_imgs
from ritnet.utils.config import get_config_from_json, setup_global_config
from ritnet.model.model_builder import build_unet_model

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
  
  config_path = "../configs/training_config/training_config_2.json"
  config = get_config_from_json(config_path)

  ##### Workaround ############
  setup_global_config(config)
  from ritnet.utils.config import GLOBAL_CONFIG
  ##### End of Workaround #####

  model_config_path = "../configs/model_config/simplenet.json"
  model_config = get_config_from_json(model_config_path)

  # Build model!
  model = build_unet_model(config, model_config, verbose=False)

  # Training
  weights_path = f"../models/{GLOBAL_CONFIG.name}_{model_config.model_name}/checkpoint"

  model.load_weights(weights_path)

  img_path = "../data/inference_data/0305.tif"

  img = np.asarray(Image.open(img_path))
  try:
    assert img.shape[-1] == 3
  except:
    if img.shape[-1] == 4:
      img = img[..., :-1]
    else:
      print("Unknown number of channel.")
      sys.exit(1)
  prep_img = preprocess_image(img, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
  
  batch_x = tf.convert_to_tensor(prep_img)[tf.newaxis, ...]
  example_id = 0

  with tf.device("/GPU:0"):
    y_pred = model(batch_x, training=False)

  print("input")
  show_img(batch_x[example_id])

  example_out = tf.nn.sigmoid(y_pred)[example_id]
  mask = tf.math.argmax(example_out, axis=-1)

  example_bg = mask == 0
  example_pupil = mask == 1
  example_iris = mask == 2
  example_sclera = mask == 3

  show_imgs([[mask, example_pupil], [example_iris, example_sclera]])
  
if __name__ == '__main__':
  main()
