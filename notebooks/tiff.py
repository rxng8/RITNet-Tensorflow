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

from ritnet.utils.utils import show_img, preprocess_image

# %%

example_path = "../data/s-general/s-general/5/mask-withskin/0001.tif"

# %%

img = Image.open(example_path)
np_img = np.asarray(img)
np_img.shape

# %%

show_img(np_img[..., 0])

# %%

iris = np_img[..., 0]

iris_region = iris > 0
np.mean(iris[iris_region])

# %%


np_img[265, 304, 2]






# %%
