# %%

import tensorflow as tf
import numpy as np
import pandas as pd

def categorical_crossentropy(target, output, from_logits=False):
  # https://github.com/keras-team/keras/blob/985521ee7050df39f9c06f53b54e17927bd1e6ea/keras/backend/numpy_backend.py#L333
  if from_logits:
    output = tf.nn.softmax(output)
  else:
    output /= output.sum(axis=-1, keepdims=True)
  output = np.clip(output, 1e-7, 1 - 1e-7)
  return np.sum(target * -np.log(output), axis=-1, keepdims=False)

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

example_out_class = 10

SEED = 123
tf.random.set_seed(SEED)
example_pred = tf.random.normal((
  example_out_class,
))

example_label = tf.convert_to_tensor([1,0,0,0,1,0,0,1,0,1], dtype=tf.float32)

# %%


loss = ce_loss(example_label, example_pred)
loss

# %%

# Square inverse of class frequency

# compute frequency of classes















