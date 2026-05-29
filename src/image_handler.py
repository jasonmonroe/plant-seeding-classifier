# image_handler.py

import random
from typing import Any, Tuple
import cv2

import numpy as np
import numpy.random
import pandas as pd

import matplotlib.pyplot as plt
from numpy import dtype, ndarray

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator 

from src.config import IMAGE_COLS, IMAGE_ROWS, GENERATOR_BATCH_SIZE, SEED


class ImageHandler:
    def __init__(self, images: np.ndarray, width: int, height: int, channels: int):
        self._images = images
        self.width = width
        self.height = height
        self.channels = channels

    def show_random_image(self, resized_images=None):
        if resized_images is None:
            imgs = self._images
        else:
            imgs = resized_images

        random_index = random.randrange(len(imgs))
        random_img = imgs[random_index]

        title = f'Random Image (Sample) | Index: {random_index}'
        plt.figure(num=title, figsize=(5, 5))
        plt.imshow(random_img, cmap='gray')
        plt.title(title)
        plt.show()

        return random_img

    def show_all_images(self):
        return self._images

    def show_image_by_index(self, index: int):
        return self._images[index]

    def show_raw_images(self, labels: np.ndarray, rows:int=IMAGE_ROWS, cols:int=IMAGE_COLS) -> None:
        """
        Visualizes random image data from the input array in a grid layout.

        Uses plt.tight_layout() to automatically adjust subplot parameters
        for a tight layout, preventing overlap.
        """

        # Assuming labels is an array where each element corresponds to an image
        fig = plt.figure(num="Raw Image Labels", figsize=(10, 8))
        label_cnt = len(labels)
        
        # Initialize the random number generator once outside the loops
        rng = np.random.default_rng(SEED)

        for i in range(cols):
            for j in range(rows):
                # Generate a unique random index for each iteration
                #random_index =  random.randint(0, label_cnt)
                random_index = rng.integers(0, label_cnt)

                # Subplots are 1-indexed (i * rows + j + 1)
                ax = fig.add_subplot(rows, cols, i * cols + j + 1)
                # Note: Changed i * rows to i * cols for standard subplot counting

                # Display the image data
                ax.imshow(self._images[random_index, :])

                # Set the title for the subplot
                ax.set_title(labels[random_index])

                # Remove the ticks/labels for a cleaner image visualization
                ax.axis('off')

        # Automatically adjust subplot parameters to give a tight layout.
        plt.tight_layout()
        plt.show()

    def create_bgr_images(self) -> ndarray[Any, dtype[Any]]:
        # You can convert the internal list/array to a numpy array directly 
        # without an explicit for loop.
        bgr_images = np.array(self._images)

        print(f'Shape of BGR images: {bgr_images.shape}')

        return bgr_images

    def show_random_cv2_image(self) -> None:

        # Using cv2_imshow to display the image
        img = random.choice(self._images)
        title = 'Random Image (OpenCV Display)'
        plt.figure(num=title, figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def convert_to_rgb(self, imgs: np.ndarray):
        # Using list comprehension for better performance and readability
        rgb_images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs])

        print(f'Shape of RGB images: {rgb_images.shape}')

        return rgb_images

    def get_resized_images(self, imgs: np.ndarray, reduced_img_dims: Tuple[int, int, int]) -> list:
        resized_images = []

        for img in imgs:
            resized_image = cv2.resize(img, reduced_img_dims, interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized_image)

        return resized_images

    def get_train_generator(self) -> ImageDataGenerator:
        # Data generator for training image data
        return ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def build_generator(
            self,
            datagen: ImageDataGenerator,
            x_training_normalized: np.ndarray,
            y_training_encoded: np.ndarray,
            shuffle_flag: bool=True
    ) -> NumpyArrayIterator:
        return datagen.flow(
            x_training_normalized,
            y_training_encoded,
            batch_size=GENERATOR_BATCH_SIZE,
            seed=SEED,
            shuffle=shuffle_flag
        )

    # Show images from training data
    def show_augmented_image_batch(self, train_generator: NumpyArrayIterator, encoder: LabelEncoder) -> None:
        img, img_labels = next(train_generator)
        fig, axes = plt.subplots(4, 4, figsize=(14, 7))
        fig.set_size_inches(12, 12)

        categories = np.unique(img_labels)
        keys = dict(enumerate(encoder.classes_))

        for (img, label_index, ax) in zip(img, np.argmax(img_labels, axis=1), axes.flatten()):
            ax.imshow(img)
            ax.set_title(keys[label_index]) # Map numeric label to plant name using keys
            ax.axis('off')

        plt.show()

    def get_resized_img_dims(self, reduce_by:int=1):
        # You can reduce the image in half but for now we will keep same size due to original images having a small filesize.
        # reduce_by = 1 # Note: was 2
        return self.height // reduce_by, self.width // reduce_by
