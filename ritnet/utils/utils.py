#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Utils file
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image


def show_img(img):
  plt.axis("off")
  plt.imshow(img)
  plt.show()

def preprocess_image(img: np.array, image_size=(320, 320)) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): Expect image with shape (height, width, 3). With range [0, 255], int
    image_size (tuple, optional): Size of the image (does not include the number of channel).
      Defaults to (320, 320).

  Returns:
    tf.Tensor: Tensor representation of the preprocessed image.
      Has shape (*image_size, img.shape[2]). Range [0, 1]
  """
  # Preprocessing image
  preprocessed_img = tf.cast(img, tf.float32)
  preprocessed_img /= 255.0
  assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
  preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
  return preprocessed_img