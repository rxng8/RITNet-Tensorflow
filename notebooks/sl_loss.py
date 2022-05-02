# %%

import numpy as np
from PIL import Image
import sys
import os
import tensorflow as tf

from scipy.ndimage import distance_transform_edt

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
from ritnet.utils.utils import preprocess_image, preprocess_label, show_img

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
zeros = tf.zeros_like(prep_label[..., 1])
a = distance_transform_edt(prep_label[..., 1], sampling=(1, GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
# a = distance_transform_edt(zeros, sampling=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

# %%

show_img(prep_label[0, ..., 1])
show_img(a[0, ...])

# %%

# prep_label = prep_label[tf.newaxis, ...]
positive_mask = tf.cast(prep_label[..., 1], tf.bool) # (batch, height, width)
negative_mask = ~positive_mask # (batch, height, width)
computation_cond = tf.reduce_any(positive_mask, axis=[1, 2]) # (batch,)
computation_cond = computation_cond[..., tf.newaxis, tf.newaxis] # (batch, 1, 1)
computation_cond = tf.broadcast_to(computation_cond, (computation_cond.shape[0], GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width)) # (batch, height, width)
computation_cond = tf.cast(computation_cond, tf.double) # (batch, height, width)

outside_distance = distance_transform_edt(negative_mask, sampling=(prep_label.shape[0], GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
inside_distance = distance_transform_edt(positive_mask, sampling=(prep_label.shape[0], GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

channel_distance_matrix = outside_distance * tf.cast(negative_mask, tf.double) - (inside_distance - 1) * tf.cast(positive_mask, tf.double) # (batch, height, width)
channel_distance_matrix = tf.convert_to_tensor(channel_distance_matrix) # (batch, height, width)

channel_distance_matrix = channel_distance_matrix * computation_cond # (batch, height, width)
channel_distance_matrix = 1e-4 * channel_distance_matrix

# %%

channel_distance_matrix

# %%



# %%

computation_cond.shape



# %%


show_img(channel_distance_matrix[0, ...])

# %%


channel_distance_matrix




# %%

## Testing

def encode_label_to_distance_matrix(label: np.ndarray) -> tf.Tensor:
  """encode_label_to_distance_matrix

  https://github.com/LIVIAETS/boundary-loss/blob/8f4457416a583e33cae71443779591173e27ec62/utils.py#L260
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
  https://arxiv.org/abs/1812.07032

  Args:
    label (np.ndarray): One-hot encoded label with muultiple channels. Expect label shape to be
      (batch_size, height, width, n_class). Label are expected to contain only zeros and ones.

  Returns:
    tf.Tensor: The distance matrix of the label as in the paper above.
  """

  assert len(label.shape) == 4

  # For every channel in the label, encode it to the distance matrix
  n_channel = label.shape[-1]
  batch_size = label.shape[0]

  res = []

  for channel in range(n_channel):
    positive_mask = tf.cast(label[..., channel], tf.bool) # (batch, height, width)
    negative_mask = ~positive_mask # (batch, height, width)
    computation_cond = tf.reduce_any(positive_mask, axis=[1, 2]) # (batch,)
    computation_cond = computation_cond[..., tf.newaxis, tf.newaxis] # (batch, 1, 1)
    computation_cond = tf.broadcast_to(computation_cond, (computation_cond.shape[0], GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width)) # (batch, height, width)
    computation_cond = tf.cast(computation_cond, tf.double) # (batch, height, width)

    outside_distance = distance_transform_edt(negative_mask, sampling=(batch_size, GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))
    inside_distance = distance_transform_edt(positive_mask, sampling=(batch_size, GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width))

    channel_distance_matrix = outside_distance * tf.cast(negative_mask, tf.double) - (inside_distance - 1) * tf.cast(positive_mask, tf.double) # (batch, height, width)
    channel_distance_matrix = tf.convert_to_tensor(channel_distance_matrix) # (batch, height, width)

    channel_distance_matrix = channel_distance_matrix * computation_cond # (batch, height, width)
    # channel_distance_matrix = loss_config.sl_dist_lambda * channel_distance_matrix
    channel_distance_matrix = 1e-4 * channel_distance_matrix
    channel_distance_matrix = channel_distance_matrix[..., tf.newaxis]
    res.append(channel_distance_matrix)

  return tf.concat(res, axis=-1)


dist_matrix = encode_label_to_distance_matrix(prep_label)
dist_matrix.shape

# %%

show_img(dist_matrix[0, ..., 3])

