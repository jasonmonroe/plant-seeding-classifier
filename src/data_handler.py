# data_handler.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import CSV_FILE, HALF_DATA_SPLIT, NPY_FILE, SEED, SOURCE_DIR, TEMPORARY_DATA_SPLIT


class DataHandler:
    """
    Loading the dataset

    Assuming DIR_PATH, CSV_FILE, and NPY_FILE are defined in the environment.

    Load the labels (target variable) from the CSV file
    This assumes the CSV file contains one column or one row per image for the label.
    We convert the labels to a flattened NumPy array immediately.
    """

    def __init__(self):
        self._source_path = Path(SOURCE_DIR)
        self._labels = self.load_data()
        self._images = self.load_images()

    def load_data(self) -> np.ndarray:
        csv_path = self._source_path / CSV_FILE
        return pd.read_csv(csv_path)

    def load_images(self) -> np.ndarray:
        """
        https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
        Load the image features (pixel data) from the NPY file.
        This is the correct function to load NumPy's binary data format.
        """

        npy_path = self._source_path / NPY_FILE
        return np.load(npy_path)

    def get_labels(self) -> pd.DataFrame:
        return self._labels

    def get_images(self) -> np.ndarray:
        return self._images

    def describe_data(self) -> None:
        labels = self._labels

        count = labels['Label'].value_counts()

        print('--- Describing Data ---')
        print(f'Plant Seedling Counts: {count}')
        print(f'Head: {labels.head()}')
        print(f'Tail: {labels.tail()}')
        print(f'Shape: {labels.shape}')
        print(f'Info: {labels.info()}')

        print('Describe')
        print(labels.describe().T)
        print(f'Null values: {labels.isnull().sum()}')

        """
        Observations:
        
        The data file (labels) has 4750 rows and 1 column.  The shape of the npy row:
        
        * batch_size: 4750 (row count)
        * height: 128
        * width: 128
        * channels: 3
        """

    def describe_label(self) -> None:
        labels = self._labels

        print('--- Describing Labels ---')
        print(f"Loaded Labels shape (Target, Y): {labels.shape}, Type: {type(labels)}")

        # Outputs: (batch_size, height, width, channels)
        print(labels.shape)

    def describe_images(self) -> tuple:
        """
        Image Information:
        mean, median, std dev, min, max
        https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """

        img = self._images

        print('--- Describing Image ---')
        print('Mean:', np.mean(img))
        print('Median:', np.median(img))
        print('Standard Deviation:', np.std(img))
        print('Minimum:', np.min(img))
        print('Maximum:', np.max(img))

        print('Printing the first element')
        #print(img[:1])  # Print the first element

        print(f"\nLoaded Images shape (Features, X): {img.shape}, Type: {type(img)}")
        print(img.shape)

        print(f'Total number of images: {img.shape[0]}')

        # Get image data
        print(f'Width:{img.shape[2]}, Height: {img.shape[1]}, RGB Channels: {img.shape[3]}')

        return img.shape[2], img.shape[1], img.shape[3]

    def split_data(self, features: np.ndarray) -> dict:
        """
        -------------------------------------------
         CREATE TRAINING, VALIDATION, TESTING DATA
        -------------------------------------------
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        Training Data ~ 80%
        Validation Data ~ 10%
        Testing Data ~ 10%

        :param features:
        :return:
        """

        # Ensure features is a numpy array for efficient slicing and processing
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        target = self._labels['Label'].to_numpy()

        # Defensive Assertion Check: Ensure absolute size alignment
        assert len(features) == len(target), f"Data length mismatch! Features: {len(features)}, Labels: {len(target)}"

        # --- Split data into 70% training data and 30% temporary data --- #
        x_train, x_temp, y_train, y_temp = train_test_split(
            features,
            target,
            test_size=TEMPORARY_DATA_SPLIT,
            random_state=SEED,
            stratify=target
        )

        # --- Next, take remaining temporary data 30% and split in half --- #
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=HALF_DATA_SPLIT,
            random_state=SEED,
            stratify=y_temp
        )

        """
        # Printing the shapes
        print('--- (Split) Data Shapes ---')
        print(f'Shape of X training: {x_train.shape}')
        print(f'Shape of Y training: {y_train.shape}')
        print(f'Shape of X validation: {x_val.shape}')
        print(f'Shape of Y validation: {y_val.shape}')
        print(f'Shape of X testing: {x_test.shape}')
        print(f'Shape of Y testing: {y_test.shape}')

        print('---(Split) Data Types ---')
        print(f'Data type of X training: {x_train.dtype}')
        print(f'Data type of Y training: {y_train.dtype}')

        # === DEBUG: RAW DATA LOADING ALIGNMENT ===
        print("### DEBUG: Verification: Initial Loading Alignment ###")
        for i in [0, 1, 2, -3, -2, -1]: # Check first 3 and last 3 items
            # Adjust 'filename_column' to match your DataFrame's image path column
            print(f"Index {i} -> File: {self._labels.iloc[i]} | Label: {self._labels.iloc[i]['Label']}")

            # If features are loaded, print statistics of the loaded array at index i
            if 'features' in locals() or 'features' in globals():
                print(f"      -> Array Shape: {features[i].shape} | Mean Pixel: {features[i].mean():.4f}")
        print("### =========================================== ####")
        # === DEBUG: RAW DATA LOADING ALIGNMENT ===
        """

        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test
        }
