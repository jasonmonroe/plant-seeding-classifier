# preprocess.py

import numpy as np
import pandas as pd

import 

from src.config import DIR_PATH, CSV_FILE, NPY_FILE, TEMPORARY_DATA_SPLIT, TRAINING_DATA_SPLIT, TESTING_DATA_SPLIT, HALF_DATA_SPLIT, SEED

def load_data():
    return pd.read_csv(DIR_PATH + CSV_FILE)


def load_images():
    return np.load(DIR_PATH + NPY_FILE)


def desc_data(df: pd.DataFrame):
    # head
    print(df.head())

    # tail
    print(df.tail())

    # shape
    print(f'Shape: {df.shape}')

    """The data file has 4750 rows and 1 column."""

    # info
    print(df.info())

    # describe T
    print(df.describe().T)

    # check for null values
    df.isnull().sum()


def desc_labels(labels, images):

    # Print the shapes to confirm successful loading
    print(f"Loaded Images shape (Features, X): {images.shape}, Type: {type(images)}")
    print(f"Loaded Labels shape (Target, Y): {labels.shape}, Type: {type(labels)}")

    # Outputs: (batch_size, height, width, channels)
    print(labels.shape)
    print(f'Total number of labels: {images.shape[0]}')

    print(images.shape)
    print(f'Total number of images: {images.shape[0]}')


def desc_images(images)
    print('Mean:', np.mean(images))
    print('Median:', np.median(images))
    print('Standard Deviation:', np.std(images))
    print('Minimum:', np.min(images))
    print('Maximum:', np.max(images))

    print("Print the first element")
    print(images[:1])  # Print the first element


def split_data(df: pd.DataFrame):
    # INDEPENDENT VARIABLES aka features
    features_df = resized_images

    # DEPENDENT VARIABLE aka target
    target_df  = df['Label']



    # --- Split data into 70% training data and 30% temporary data --- #
    x_training_data, x_temp_data, y_training_data, y_temp_data = train_test_split(
        features_df,
        target_df,
        test_size=TEMPORARY_DATA_SPLIT,
        random_state=SEED,
        stratify=target_df
    )

    # --- Then take remaining temporary data 30% and split in half --- #
    x_validation_data, x_testing_data, y_validation_data, y_testing_data = train_test_split(
        x_temp_data,
        y_temp_data,
        test_size=HALF_DATA_SPLIT,
        random_state=SEED,
        stratify=y_temp_data
    )

    return x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data
