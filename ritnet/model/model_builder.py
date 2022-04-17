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
import tensorflow.python.keras.layers as layers

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
    
    if conv['stride'] > 1: x = layers.ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
    x = layers.Conv2D(
      conv['filter'], 
      conv['kernel'], 
      strides=conv['stride'], 
      padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
      name='conv_' + str(conv['layer_idx']), 
      use_bias=False if conv['bnorm'] else True
    )(x)
    if conv['bnorm']: x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['leaky']: x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

  return layers.add([skip_connection, x]) if skip else x

def perform_upsampling():
  pass

def perform_concatenate():
  pass

def perform_max_pool():
  pass

def build_unet_model(model_config: Munch) -> tf.python.keras.models.Model:

  pass
  
  