#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 28, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

File contains loss functions
"""

import os
from munch import Munch
import tensorflow as tf
import numpy as np
from typing import List

from ..utils.config import GLOBAL_CONFIG

def get_loss_func_by_loss_config(loss_config: Munch):
  loss_func = None
  if "gdl" in loss_config.name and "bal" in loss_config.name and "sl" in loss_config.name:
    loss_func = get_gdl_bal_sl_loss_func(loss_config)
  elif "gdl" in loss_config.name and "bal" in loss_config.name and "sl" not in loss_config.name:
    loss_func = get_gdl_bal_loss_func(loss_config)
  elif "gdl" not in loss_config.name and "bal" in loss_config.name and "sl" in loss_config.name:
    loss_func = get_bal_sl_loss_func(loss_config)
  elif "gdl" in loss_config.name and "bal" not in loss_config.name and "sl" in loss_config.name:
    loss_func = get_gdl_sl_loss_func(loss_config)
  elif "gdl" in loss_config.name and "bal" not in loss_config.name and "sl" not in loss_config.name:
    loss_func = get_gdl_loss_func(loss_config)
  elif "gdl" not in loss_config.name and "bal" in loss_config.name and "sl" not in loss_config.name:
    loss_func = get_bal_loss_func(loss_config)
  elif "gdl" not in loss_config.name and "bal" not in loss_config.name and "sl" in loss_config.name:
    loss_func = get_sl_loss_func(loss_config)
  else:
    loss_func = get_normal_ce_loss_func(loss_config)
  return loss_func

def get_gdl_bal_sl_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_gdl_bal_sl_loss")
  pass

def get_gdl_bal_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_gdl_bal_loss")
  pass 

def get_bal_sl_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_bal_sl_loss")
  pass 

def get_gdl_sl_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_gdl_sl_loss")
  pass 

def get_gdl_loss_func(loss_config: Munch, verbose=True):

  if verbose:
    print("[Loss] get_gdl_loss")

  def gdl_loss(true, pred):

    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(true, pred)

    sigmoid_pred = tf.nn.sigmoid(pred)

    flat_true = tf.reshape(true, (GLOBAL_CONFIG.batch_size, -1, GLOBAL_CONFIG.n_class))
    flat_pred = tf.reshape(sigmoid_pred, (GLOBAL_CONFIG.batch_size, -1, GLOBAL_CONFIG.n_class))

    invariance_per_class = 1 / (tf.pow(tf.reduce_sum(flat_true, axis=-2), 2) + 1e-8)

    multiple = tf.reduce_sum(flat_true * flat_pred, axis=-2)
    summa = tf.reduce_sum(flat_true + flat_pred, axis=-2)

    numer = tf.reduce_sum(invariance_per_class * multiple, axis=-1)
    denom = tf.reduce_sum(invariance_per_class * summa, axis=-1)

    gdl_b = 1 - 2 * tf.divide(numer, denom + 1e-8)

    gdl = tf.reduce_sum(gdl_b)

    # if verbose:
    #   print(pred)
    #   print(invariance_per_class)

    return ce_loss + loss_config.theta * gdl
  
  return gdl_loss

def get_bal_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_bal_loss")
  pass 

def get_sl_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_sl_loss")
  pass 

def get_normal_ce_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_normal_ce_loss")
  return tf.keras.losses.CategoricalCrossentropy(from_logits=True)