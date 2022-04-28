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

example_path = "../data/s-general/8/synthetic/0001.tif"
example_path2 = "../data/s-general/8/mask-withskin/0001.tif"

img = Image.open(example_path2)
np_img = np.asarray(img)
np_img.shape

# show_img(np_img[...])
