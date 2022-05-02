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

  def ce_gdl_sl(true, pred, dist_matrix):
    cel = ce_loss(true, pred)
    gdl = gdl_loss(loss_config, true, pred)
    surface_loss = sl_loss(loss_config, pred, dist_matrix)
    return cel + gdl + surface_loss

  return ce_gdl_sl

def get_gdl_loss_func(loss_config: Munch, verbose=True):

  if verbose:
    print("[Loss] get_gdl_loss")

  def ce_gdl_loss(true, pred):
    cel = ce_loss(true, pred)
    gdl = gdl_loss(loss_config, true, pred)
    return cel + gdl
  
  return ce_gdl_loss

def get_bal_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_bal_loss")
  pass 

def get_sl_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_sl_loss")

  def ce_sl_loss(true, pred, dist_matrix):
    cel = ce_loss(true, pred)
    surface_loss = sl_loss(loss_config, pred, dist_matrix)
    
    return cel + surface_loss

  return ce_sl_loss

def get_normal_ce_loss_func(loss_config: Munch, verbose=True):
  if verbose:
    print("[Loss] get_normal_ce_loss")
  return ce_loss


## Loss Algorithms ##

def ce_loss(true, pred):
  return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(true, pred)

def gdl_loss(loss_config, true, pred):
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

  return loss_config.gdl_theta * gdl

def sl_loss(loss_config, pred, dist_matrix):
  assert pred.shape == dist_matrix.shape
  sigmoid_pred = tf.nn.sigmoid(pred)
  surface_loss = tf.reduce_mean(sigmoid_pred * dist_matrix)
  return loss_config.sl_theta * surface_loss