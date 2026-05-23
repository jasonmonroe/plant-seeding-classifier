# image_handler.py

import random
from typing import Any, Tuple
import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from numpy import dtype, ndarray

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator 

from src.config import IMAGE_PX_MAX, IMAGE_COLS, IMAGE_ROWS, GENERATOR_BATCH_SIZE, SEED


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

        random_index = random.randint(0, len(imgs))
        random_img = imgs[random_index]

        plt.imshow(random_img, cmap='gray')
        plt.title(f'Random Image (Sample) | Index: {random_index}')
        plt.show()

        return random_img

    def show_all_images(self):
        return self._images

    def show_image_by_index(self, index: int):
        return self._images[index]

    def show_raw_images(self, labels: np.darray, rows:int=IMAGE_ROWS, cols:int=IMAGE_COLS) -> None:
        """
        Visualizes random image data from the input array in a grid layout.

        Uses plt.tight_layout() to automatically adjust subplot parameters
        for a tight layout, preventing overlap.
        """

        # Assuming labels is an array where each element corresponds to an image
        fig = plt.figure(figsize=(10, 8))
        label_cnt = len(labels)

        for i in range(cols):
            for j in range(rows):
                # Select a random index from the available labels/images
                # We use label_cnt as the upper bound (exclusive)
                random_index = np.random.randint(0, label_cnt)

                # Subplots are 1-indexed (i * rows + j + 1)
                ax = fig.add_subplot(rows, cols, i * cols + j + 1) # Note: Changed i * rows to i * cols for standard subplot counting

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
        bgr_images = []
        for img in self._images:
            bgr_images.append(img)

        bgr_images = np.array(bgr_images)

        # Now, BGR images contains all images...
        print(f'Shape of BGR images: {bgr_images.shape}')

        return bgr_images

    def show_random_cv2_image(self) -> None:
        # Using cv2_imshow to display the image
        img = random.choice(self._images)
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Random Image (OpenCV Display)')
        plt.axis('off')
        plt.show()

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def convert_to_rgb(self, imgs: np.ndarray):
        rgb_images = []
        for img in imgs:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)

        # Get image RGB
        rgb_images = np.array(rgb_images)

        print(f'Shape of RGB images: {rgb_images.shape}')

        return rgb_images

    def get_resized_images(self, imgs: np.ndarray, reduced_img_dims: Tuple[int, int, int]) -> list:
        resized_images = []

        for img in imgs:
            resized_image = cv2.resize(img, reduced_img_dims, interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized_image)

        return resized_images

    def is_resized(self, img, dims) -> bool:
        self.height, self.width = img.shape[:2]
        print(f'Resized Image: Width: {self.width}, Height: {self.height}')

        # Check if resized worked: Get height, width of image
        if self.height == dims[0] and self.width == dims[1]:
            print(f'Image resized successfully.  Updating Image: {self.width} x {self.height}, dims & params...\n')
            self.show_random_cv2_image()

            return True

        print('Image resizing failed!')

        return False

    def generate_training_image_batch(self):
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
            data_gen: ImageDataGenerator,
            x_training_normalized: np.ndarray,
            y_training_encoded: np.ndarray,
            shuffle_flag: bool=True
    ) -> NumpyArrayIterator:
        return data_gen.flow(
            x_training_normalized,
            y_training_encoded,
            batch_size=GENERATOR_BATCH_SIZE,
            seed=SEED,
            shuffle=shuffle_flag
        )

    # Show images from training data
    def show_augmented_image_batch(train_generator: NumpyArrayIterator, encoder: LabelEncoder) -> None:
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
