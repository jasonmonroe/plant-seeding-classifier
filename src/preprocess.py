# @todo DELETE
# preprocess.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import NPY_FILE, SOURCE_DIR, CSV_FILE, NPY_FILE, TEMPORARY_DATA_SPLIT, HALF_DATA_SPLIT, SEED

def load_data():
    return pd.read_csv(SOURCE_DIR + CSV_FILE)

def load_images():
    return np.load(SOURCE_DIR + NPY_FILE)


def describe_data(df: pd.DataFrame):
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


def describe_labels(labels, images):

    # Print the shapes to confirm successful loading
    
    print(f"Loaded Labels shape (Target, Y): {labels.shape}, Type: {type(labels)}")

    # Outputs: (batch_size, height, width, channels)
    print(labels.shape)
    print(f'Total number of labels: {images.shape[0]}')



def describe_images(images):
    print('Mean:', np.mean(images))
    print('Median:', np.median(images))
    print('Standard Deviation:', np.std(images))
    print('Minimum:', np.min(images))
    print('Maximum:', np.max(images))

    print("Print the first element")
    print(images[:1])  # Print the first element

    print(f"Loaded Images shape (Features, X): {images.shape}, Type: {type(images)}")

    print(images.shape)
    print(f'Total number of images: {images.shape[0]}')


def split_data(df_features: pd.DataFrame, df_target):
    # INDEPENDENT VARIABLES aka features
    # df are the resized images
    #df_features = df

    # DEPENDENT VARIABLE aka target
    #df_target = df_label


    # --- Split data into 70% training data and 30% temporary data --- #
    x_training_data, x_temp_data, y_training_data, y_temp_data = train_test_split(
        df_features,
        df_target,
        test_size=TEMPORARY_DATA_SPLIT,
        random_state=SEED,
        stratify=df_target
    )

    # --- Then take remaining temporary data 30% and split in half --- #
    x_validation_data, x_testing_data, y_validation_data, y_testing_data = train_test_split(
        x_temp_data,
        y_temp_data,
        test_size=HALF_DATA_SPLIT,
        random_state=SEED,
        stratify=y_temp_data
    )

    # Printing the shapes
    print('Data Shapes')

    # Convert lists to NumPy arrays before checking shape and dtype
    x_training_data = np.array(x_training_data)
    y_training_data = np.array(y_training_data)
    x_validation_data = np.array(x_validation_data)
    y_validation_data = np.array(y_validation_data)
    x_testing_data = np.array(x_testing_data)
    y_testing_data = np.array(y_testing_data)

    print(f'Shape of X training: {x_training_data.shape}')
    print(f'Shape of Y training: {y_training_data.shape}')
    print(f'Shape of X validation: {x_validation_data.shape}')
    print(f'Shape of Y validation: {y_validation_data.shape}')
    print(f'Shape of X testing: {x_testing_data.shape}')
    print(f'Shape of Y testing: {y_testing_data.shape}')

    print('Data Types')
    print(f'Data type of X training: {x_training_data.dtype}')
    print(f'Data type of Y training: {y_training_data.dtype}')

    return x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data
