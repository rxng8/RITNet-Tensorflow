# %%

import numpy as np
from PIL import Image
import sys
import os
import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt

print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import argparse

from ritnet.utils.config import get_config_from_json, setup_global_config

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

config_path = "../configs/training_config/training_config_2.json"
model_config_path = "../configs/model_config/simplenet.json"

config = get_config_from_json(config_path)
model_config = get_config_from_json(model_config_path)

##### Workaround to setup global config ############
setup_global_config(config)
from ritnet.utils.config import GLOBAL_CONFIG
##### End of Workaround #####

# Because the generator and some classes are based on the
# GLOBAL_CONFIG, we have to import them after we set the config
from ritnet.model.model_builder import build_unet_model
from ritnet.utils.utils import preprocess_image, preprocess_label, show_img, show_imgs

# Define path
image_path = "../data/s-general/7/synthetic/0011.tif"
label_path = "../data/s-general/7/mask-withskin/0011.tif"

# Get image from path
image = np.asarray(Image.open(image_path))
label = np.asarray(Image.open(label_path))

# Workaround for inconsistent data. Some input data has 3 channels,
# while some of them has 4 channels with an empty channels in the end.
# In that case we just remove the last channel.
try:
  assert image.shape[-1] == 3
except:
  if image.shape[-1] == 4:
    image = image[..., :-1]
  else:
    print("Unknown number of channel.")
    sys.exit(1)
# Do the same check for labels.
assert label.shape[-1] == 3, "Inconsistent label."

prep_image = preprocess_image(image, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))[tf.newaxis, ...]
prep_label = preprocess_label(label, image_size=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))[tf.newaxis, ...]


# %%

img = tf.cast(prep_label[0, ...], tf.uint8) * 255
img = img.numpy()
edges_whole = cv.Canny(img[..., 0], 100, 200)
edges_pupil = cv.Canny(img[..., 1], 100, 200)
edges_iris = cv.Canny(img[..., 2], 100, 200)
edges_sclera = cv.Canny(img[..., 3], 100, 200)

show_imgs([[edges_whole, edges_pupil], [edges_iris, edges_sclera]])


# %%

show_img(prep_image[0, ...])