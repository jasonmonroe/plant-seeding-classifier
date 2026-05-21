# imaging.py

import random
import cv2

import numpy as np
import matplotlib.pyplot as plt

from google.colab.patches import cv2_imshow

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    NumpyArrayIterator # The iterator is generally created by ImageDataGenerator
)

from src.config import GENERATOR_BATCH_SIZE, IMAGE_COLS, IMAGE_PX_MAX, IMAGE_ROWS, SEED # OpenCV for image processing


def show_random_cv2_image(imgs: np.ndarray) -> None:
    cv2_imshow(random.choice(imgs)) # Using cv2_imshow to display the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def show_random_image(resized_images: list) -> None:
def is_resized(img, resized_img_dims) -> bool:
    #random_img = random.choice(resized_images)
    height, width = img.shape[:2]

    print(f'Resized Image: Width: {width}, Height: {height}')
    
    # Check if resized worked: Get height, width of image
    if height == resized_img_dims[0] and width == resized_img_dims[1]:
        print('Image resized successfully.  Updating Image: height, width, dims, params...\n')
   
        #IMAGE_HEIGHT = REDUCED_IMAGE_DIMS[0]
        #IMAGE_WIDTH = REDUCED_IMAGE_DIMS[1]
        #IMAGE_DIMS = REDUCED_IMAGE_DIMS
        #IMAGE_PARAMS = IMAGE_DIMS + (IMAGE_CHANNELS, )

        show_random_cv2_image(img)

        return True
    
    print('Image resizing failed.')
    return False


def create_bgr_images(imgs: np.ndarray) -> list:
    bgr_images = []
    for img in imgs:
        bgr_images.append(img)

    bgr_images = np.array(bgr_images)

    # Now, BGR images contains all images...
    print(f'Shape of BGR images: {bgr_images.shape}')

    return bgr_images


def convert_to_rgb(imgs: np.ndarray) -> list:
    rgb_images = []
    for img in imgs:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb_image)

    # Get image RGB
    rgb_images = np.array(rgb_images)

    print(f'Shape of RGB images: {rgb_images.shape}')

    return rgb_images


# Show random image using matplotlib
def show_random_image(imgs: np.ndarray):
    random_index = random.randint(0, len(imgs))
    random_img = imgs[random_index]
    plt.imshow(random_img, cmap='gray')
    plt.title(f'Random Image (Sample) | Index: {random_index}')
    plt.show()

    return random_img


def show_raw_images(imgs: np.ndarray, labels: np.ndarray, rows:int=IMAGE_ROWS, cols:int=IMAGE_COLS) -> None:
    """
    Visualizes random image data from the input array in a grid layout.

    Uses plt.tight_layout() to automatically adjust subplot parameters
    for a tight layout, preventing overlap.
    """
    fig = plt.figure(figsize=(10, 8))
    label_cnt = len(labels) # Assuming labels is an array where each element corresponds to an image

    for i in range(cols):
        for j in range(rows):
            # Select a random index from the available labels/images
            # We use label_cnt as the upper bound (exclusive)
            random_index = np.random.randint(0, label_cnt)

            # Subplots are 1-indexed (i * rows + j + 1)
            ax = fig.add_subplot(rows, cols, i * cols + j + 1) # Note: Changed i * rows to i * cols for standard subplot counting

            # Display the image data
            ax.imshow(imgs[random_index, :])

            # Set the title for the subplot
            ax.set_title(labels[random_index])

            # Remove the ticks/labels for a cleaner image visualization
            ax.axis('off')

    # Automatically adjust subplot parameters to give a tight layout.
    plt.tight_layout()

    plt.show()


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


def get_resized_images(imgs: np.ndarray, REDUCED_IMAGE_DIMS) -> list:
    resized_images = []

    for img in imgs:
        resized_image = cv2.resize(img, REDUCED_IMAGE_DIMS, interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized_image)

    return resized_images


def generate_training_image_batch():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def build_generator(data_gen: ImageDataGenerator, x_training_normalized: np.ndarray, y_training_encoded: np.ndarray, shuffle_flag: bool=True) -> NumpyArrayIterator:
    return data_gen.flow(
        x_training_normalized,
        y_training_encoded,
        batch_size=GENERATOR_BATCH_SIZE,
        seed=SEED,
        shuffle=shuffle_flag
    )


# Normalize the image(s)
def normalize(img: np.ndarray) -> float:
    return img.astype('float32') / IMAGE_PX_MAX
