# %%

import tensorflow as tf
import numpy as np
import pandas as pd

# SEED = 123
# tf.random.set_seed(SEED)

def categorical_crossentropy(target, output, from_logits=False):
  # https://github.com/keras-team/keras/blob/985521ee7050df39f9c06f53b54e17927bd1e6ea/keras/backend/numpy_backend.py#L333
  if from_logits:
    output = tf.nn.softmax(output)
  else:
    output /= output.sum(axis=-1, keepdims=True)
  output = np.clip(output, 1e-7, 1 - 1e-7)
  return np.sum(target * -np.log(output), axis=-1, keepdims=False)

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

batch_size = 2
n_feature = 4
out_class = 5


pred = tf.nn.sigmoid(
  tf.random.normal((
    batch_size,
    n_feature,
    out_class,
  ))
)

label = tf.convert_to_tensor([[[1,0,0,0,1],
                              [1,1,0,1,0],
                              [0,0,1,0,1],
                              [1,0,0,1,0]],
                              
                              [[1,0,1,0,1],
                              [0,0,0,0,0],
                              [1,0,1,0,0],
                              [0,1,0,0,1]]], dtype=tf.float32)

# label = tf.ones_like(label)

# loss = ce_loss(label, pred)
# loss

# %%



