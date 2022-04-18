#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

File contain model
"""

from typing import Dict, List
from munch import Munch
import tensorflow as tf
import tensorflow.keras.layers as layers

def perform_conv_blocks(inp: tf.Tensor, convs: List[Dict[str, int]], skip: bool=False) -> tf.Tensor:
  """Perform multiple convolution blocks specified in convs.

  Args:
    inp (tf.Tensor): input tensor
    convs (List[Dict[str, int]]): Convolution performing info. Example of a convs can be:
      [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
      {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
      {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}]
    skip (bool, optional): Whether to do a skip connection as in ResNet. Defaults to True.

  Returns:
    tf.Tensor: The tensor which has been loaded through those convolutional blocks.
  """
  x = inp
  count = 0
  
  for conv in convs:
    if count == (len(convs) - 2) and skip:
      skip_connection = x
    count += 1
    
    # if conv['stride'] > 1: x = layers.ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
    x = layers.Conv2D(
      conv['filter'], 
      conv['kernel'], 
      strides=conv['stride'], 
      padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
      name='conv_' + str(conv['layer_idx']), 
      use_bias=False if conv['bnorm'] else True
    )(x)
    if conv['bnorm']: x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if 'dropout' in conv: x = layers.Dropout(conv['dropout'], name=f"dropout_{conv['layer_idx']}")(x)
    if conv['leaky']: x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

  return layers.add([skip_connection, x]) if skip else x

def perform_upsampling(inp: tf.Tensor, ups: Munch):
  """Perform upsampling on input tensor

  Args:
    inp (tf.Tensor): _description_
    ups (Dict[str, int]): _description_
  """
  return layers.UpSampling2D(size=ups.size, name=f"upsampling_{ups.layer_idx}")(inp)

def perform_conv_transpose_block(inp: tf.Tensor, conv: Munch) -> tf.Tensor:
  """_summary_

  More resource: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

  Args:
    inp (tf.Tensor): _description_
    conv (Munch): _description_

  Returns:
    tf.Tensor: _description_
  """
  x = inp
  x = layers.Conv2DTranspose(conv.filter, conv.kernel, strides=conv.stride, padding='same', use_bias=False, name=f"trans_conv_{conv.layer_idx}")(x)
  if conv.bnorm: x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv.layer_idx))(x)
  if "dropout" in conv: x = layers.Dropout(conv.dropout, name=f"dropout_{conv.layer_idx}")(x)
  if conv.leaky: x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
  return x

def perform_concatenate(inp1: tf.Tensor, inp2: tf.Tensor, concat: Munch):
  """_summary_

  Args:
    inp1 (tf.Tensor): _description_
    inp2 (tf.Tensor): _description_
    concat (Munch): _description_

  Returns:
    _type_: _description_
  """
  return layers.Concatenate(name=f"concat_{concat.layer_idx}")([inp1, inp2])

def perform_max_pool(inp: tf.Tensor, max_pool: Munch):
  """_summary_

  Args:
    inp (tf.Tensor): _description_
    max_pool (Munch): _description_
  """
  return layers.MaxPool2D(pool_size=max_pool.pool_size, strides=max_pool.stride, name=f"max_pool_{max_pool.layer_idx}")(inp)

def build_unet_model(general_config: Munch, model_config: Munch) -> tf.keras.models.Model:
  """Build an unet model based on configuurations

  Args:
    general_config (Munch): _description_
    model_config (Munch): _description_

  Returns:
    tf.keras.models.Model: _description_
  """
  input_tensor = layers.Input(shape=(
    general_config.image_size.height, 
    general_config.image_size.width, 
    general_config.channel
  ))

  for layer_dict in model_config.model:
    print(layer_dict.name)
    pass  

  
  
  