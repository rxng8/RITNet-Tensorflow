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

from ritnet.trainer.trainer import Trainer
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
  
  config_path = "../configs/general_config.json"
  config = get_config_from_json(config_path)

  ##### Workaround ############
  setup_global_config(config)
  from ritnet.utils.config import GLOBAL_CONFIG
  ##### End of Workaround #####

  model_config_path = "../configs/simplenet.json"
  model_config = get_config_from_json(model_config_path)

  # Because the generate is based on the GLOBAL_CONFIG, we have to import them after we set the config
  from ritnet.dataloader.dataloader import train_generator, test_generator

  # Data loader
  train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
      tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32)
    )
  )
  train_batch_dataset = train_dataset.batch(GLOBAL_CONFIG.batch_size)
  train_batch_iter = iter(train_batch_dataset)

  test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
      tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32)
    )
  )
  test_batch_dataset = test_dataset.batch(GLOBAL_CONFIG.batch_size)
  test_batch_iter = iter(test_batch_dataset)

  # Build model!
  model = build_unet_model(config, model_config, verbose=True)

  # Training
  optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
  history_path = f"../history/history_{model_config.model_name}.npy"
  weights_path = f"../models/{model_config.model_name}/checkpoint"

  trainer = Trainer(
    model,
    train_batch_iter,
    test_batch_iter,
    optimizer,
    tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    epochs=1,
    steps_per_epoch=5, # 34000 // 4
    valid_step=5,
    history_path=history_path,
    weights_path=weights_path,
    save_history=True
  )

  history = trainer.train()

if __name__ == '__main__':
  main()
