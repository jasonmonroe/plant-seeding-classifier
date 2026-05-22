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






# Show images from training data
def show_augmented_image_batch(train_generator: NumpyArrayIterator, enc: LabelEncoder) -> None:
    img, img_labels = next(train_generator)
    fig, axes = plt.subplots(4, 4, figsize=(14, 7))
    fig.set_size_inches(12, 12)

    categories = np.unique(img_labels)
    keys = dict(enumerate(enc.classes_))

    for (img, label_index, ax) in zip(img, np.argmax(img_labels, axis=1), axes.flatten()):
        ax.imshow(img)
        ax.set_title(keys[label_index]) # Map numeric label to plant name using keys
        ax.axis('off')

    plt.show()










# Normalize the image(s)
def normalize(img: np.ndarray) -> float:
    return img.astype('float32') / IMAGE_PX_MAX
