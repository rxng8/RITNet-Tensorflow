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

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import cv2
import tqdm

import tensorflow as tf

from ritnet.utils.utils import show_img, preprocess_image

example_path = "../data/s-general/s-general/5/mask-withskin/0001.tif"

img = Image.open(example_path)
np_img = np.asarray(img)
np_img.shape

np_img = tf.image.resize(np_img, (192, 256), method=tf.image.ResizeMethod.BILINEAR)

# %%


np_img

# %%

tf.reduce_max(np_img)

# %%

show_img(np_img[..., 0])

# %%

iris = np_img[..., 0]

iris_region = iris > 0
np.mean(iris[iris_region])

# %%


np_img[265, 304, 2]


# %%

mask = tf.math.argmax(np_img, axis=-1)
mask.shape

# %%

show_img(mask == 0)


# %%

eye_mask = tf.cast(
  tf.reduce_any(tf.cast(np_img, tf.bool), axis=-1),
  tf.float32
)[..., tf.newaxis]

background_mask = 1 - eye_mask

pupil_mask = eye_mask * tf.cast(
  tf.math.argmax(np_img, axis=-1) == 0,
  tf.float32
)[..., tf.newaxis]

iris_mask = eye_mask * tf.cast(
  tf.math.argmax(np_img, axis=-1) == 1,
  tf.float32
)[..., tf.newaxis]

sclera_mask = eye_mask * tf.cast(
  tf.math.argmax(np_img, axis=-1) == 2,
  tf.float32
)[..., tf.newaxis]


# %%
background_mask.shape
background_mask
# %%

show_img(background_mask)
show_img(pupil_mask)
show_img(iris_mask)
show_img(sclera_mask)

# %%

res = tf.concat([background_mask, pupil_mask, iris_mask, sclera_mask], axis=-1)

# %%

res.shape

# %%

show_img(res[..., 3])


# %%
t = tf.convert_to_tensor([3, 4, 0, 1, 2, 0], dtype=tf.float32)
e = tf.cast(
  tf.cast(t, tf.bool),
  tf.float32
)
e

