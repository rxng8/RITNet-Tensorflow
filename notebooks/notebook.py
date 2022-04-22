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

from ritnet.utils.utils import show_img, preprocess_image
from ritnet.utils.config import get_config_from_json
from ritnet.model.model_builder import build_unet_model

MODEL_ARCHITECTURE_FOLDER = "../src/model/"
config = get_config_from_json("../configs/general_config.json")


# %%

# https://www.tensorflow.org/guide/data_performance
def train_generator():

  dataset_root = "../data/s-general/s-general"
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

def test_generator():
  dataset_root = "../data/s-general/s-general"
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
    tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32),
    tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32)
  ),
)
train_batch_dataset = train_dataset.batch(config.batch_size)
train_batch_iter = iter(train_batch_dataset)
# example_image, example_label = next(train_batch_iter)

test_dataset = tf.data.Dataset.from_generator(
  test_generator,
  output_signature=(
    tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32),
    tf.TensorSpec(shape=(config.image_size.height, config.image_size.width, config.channel), dtype=tf.float32)
  ),
)
test_batch_dataset = test_dataset.batch(config.batch_size)
test_batch_iter = iter(test_batch_dataset)

# Debugging

# example_label_path = "../data/s-general/s-general/23/mask-withskin/0000.tif"
# example_label = np.asarray(Image.open(example_label_path))
# example_label.shape

# example_image_path = "../data/s-general/s-general/23/synthetic/0000.tif"
# example_image = np.asarray(Image.open(example_image_path))

# multiply = preprocess_image(example_image) * preprocess_image(example_label)
# show_img(multiply)

# %%

from ritnet.model.model_builder import build_unet_model
from ritnet.utils.config import get_config_from_json

config = get_config_from_json("../configs/general_config.json")
# model_config = get_config_from_json("../configs/unet1.json")
model_config = get_config_from_json("../configs/simplenet.json")
# model_config = get_config_from_json("../configs/really_simple.json")

model = build_unet_model(config, model_config, verbose=True)

# %%

## Training code


def train_step(batch_x, batch_label, model, loss_function, optimizer):
  with tf.device("/CPU:0"):
    with tf.GradientTape() as tape:
      logits = model(batch_x, training=True)
      loss = loss_function(batch_label, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

def train(model, 
        training_batch_iter, 
        test_batch_iter, 
        optimizer, 
        loss_function,
        epochs=1, 
        steps_per_epoch=20, 
        valid_step=5,
        history_path=None,
        weights_path=None,
        save_history=False):
  
  if history_path != None and os.path.exists(history_path):
    # Sometimes, we have not created the files
    with open(history_path, "rb") as f:
      history = np.load(f, allow_pickle=True)
    epochs_loss, epochs_val_loss = history
    epochs_loss = epochs_loss.tolist()
    epochs_val_loss = epochs_val_loss.tolist()
  else:
    epochs_val_loss = []
    epochs_loss = []
  
  if weights_path != None and os.path.exists(weights_path + ".index"):
    try:
      model.load_weights(weights_path)
      print("Model weights loaded!")
    except:
      print("cannot load weights!")

  for epoch in range(epochs):
    losses = []

    with tf.device("/CPU:0"):
      step_pointer = 0
      while step_pointer < steps_per_epoch:
        batch = next(training_batch_iter)
        batch_x = batch[0]
        batch_label = batch[1]
        loss = train_step(batch_x, batch_label, model, loss_function, optimizer, step=step_pointer + 1)
        print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
        losses.append(loss)

        if (step_pointer + 1) % valid_step == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step_pointer + 1, float(loss))
          )
          # perform validation
          val_batch = next(test_batch_iter)
          logits = model(val_batch[0], training=False)
          val_loss = loss_function(val_batch[1], logits)
          print(f"exmaple logits: {logits}")
          print(f"Validation loss: {val_loss}\n-----------------")
        if (step_pointer + 1) == steps_per_epoch:
          val_batch = next(test_batch_iter)
          logits = model(val_batch[0], training=False)
          val_loss = loss_function(val_batch[1], logits)
          epochs_val_loss.append(val_loss)

        step_pointer += 1
    epochs_loss.append(losses)

    # Save history and model
    if history_path != None and save_history:
      np.save(history_path, [epochs_loss, epochs_val_loss])
    
    if weights_path != None:
      model.save_weights(weights_path)
  
  # return history
  return [epochs_loss, epochs_val_loss]

# def loss

# %%

optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)


history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  epochs=1,
  steps_per_epoch=100, # 1800 // 16
  valid_step=20,
  history_path=None,
  weights_path=None,
  save_history=False
)
