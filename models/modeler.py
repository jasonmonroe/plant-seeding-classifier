# models/modeler.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from sklearn.metrics import (
    classification_report
)
from src.config import IMAGE_PX_MAX


class Modeler:
    def __init__(self, dataset: dict):
        self._encoder = dataset.get('_encoder')
        self.plant_species = []

        # Data variables
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

        # Encoded variables
        self.y_train_enc = []
        self.y_val_enc = []
        self.y_test_enc = []

        # Normalized variables
        self.x_train_norm = []
        self.x_val_norm = []
        self.x_test_norm = []
        self.y_test_norm = []

        # Safely inject dataset values
        self.set_dataset(dataset)

    @property
    def plant_species_cnt(self) -> int:
        # Dynamically return the count of plant species.
        return len(self.plant_species) if self.plant_species else 0

    def set_dataset(self, dataset: dict) -> None:
        """
        Dynamically sets attributes and synchronizes the plant_species list
        with the internal LabelEncoder to prevent mapping drift.
        """
        for key, value in dataset.items():
            # Allow plant_species to be set normally from the incoming dict
            if hasattr(self, key) and key != 'plant_species_cnt':
                setattr(self, key, value)

        # Force self.plant_species to match the encoder's internal mapping.
        # This guarantees that Index 0 in the matrix ALWAYS matches Index 0 in text fields.
        if hasattr(self._encoder, 'classes_'):
            self.plant_species = list(self._encoder.classes_)

    def get_proc_dataset(self) -> dict:
        """
        Get processed data that has been encoded and normalized.
        :return: dict of processed data.
        """
        return {
            '_encoder': self._encoder,
            'plant_species': self.plant_species,
            'y_train_enc': self.y_train_enc,
            'x_train_norm': self.x_train_norm,
            'y_test_enc': self.y_test_enc,
            'x_test_norm': self.x_test_norm,
            'y_val_enc': self.y_val_enc,
            'x_val_norm': self.x_val_norm
        }

    def encode_data(self) -> None:
        """
        https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical

        Used to one-hot encode integer labels into a binary matrix representation that's
        suitable for classification tasks in deep learning.
        """

        self.y_train_enc = self._encode_label(self.y_train)
        self.y_test_enc = self._encode_label(self.y_test)
        self.y_val_enc = self._encode_label(self.y_val)

    def _encode_label(self, y_data) -> np.ndarray:

        # Generate the local one-hot encoded result first
        y_data = self.__convert(y_data)

        encoded_result = tf.keras.utils.to_categorical(
            self._encoder.transform(y_data),
            num_classes=self.plant_species_cnt
        )

        return encoded_result

    def __convert(self, y_data) -> list:
        """
        Safely flattens string targets regardless of whether they are
        passed as Pandas DataFrames, Series, or 1D NumPy arrays.
        Convert to a clean, flat 1D list of raw strings safely
        """

        if hasattr(y_data, 'values'):
            # If it's a Pandas object
            return list(y_data.values.ravel())
        elif isinstance(y_data, np.ndarray):
            # If it's already a NumPy array, ensure it is completely flattened
            return list(y_data.ravel())
        else:
            # Fallback for standard lists
            return list(y_data)

    def normalize(self) -> None:
        # Scaling numerical features into a standardized range, typically 0 to 1 or -1 to 1.
        self.x_train_norm = self.x_train.astype('float32') / IMAGE_PX_MAX
        self.x_test_norm = self.x_test.astype('float32') / IMAGE_PX_MAX
        self.x_val_norm = self.x_val.astype('float32') / IMAGE_PX_MAX

    def print_classification_report(
            self,
            model,
            x_data: np.ndarray,
            y_true_encoded: np.ndarray
    ) -> None:
        y_true_labels = np.argmax(y_true_encoded, axis=1)
        y_pred_probs = model.predict(x_data)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print(classification_report(
            y_true_labels,
            y_pred_classes,
            target_names=self.plant_species,
            digits=4)
        )
