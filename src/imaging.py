# @todo - DELETE
# imaging.py

import random
import cv2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Google Colab -> from google.colab.patches import cv2_imshow

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    NumpyArrayIterator # The iterator is generally created by ImageDataGenerator
)

from src.config import GENERATOR_BATCH_SIZE, IMAGE_COLS, IMAGE_PX_MAX, IMAGE_ROWS, SEED # OpenCV for image processing













# Show random image using matplotlib
















# Normalize the image(s)
#def normalize(img: np.ndarray) -> float:
#    return img.astype('float32') / IMAGE_PX_MAX
