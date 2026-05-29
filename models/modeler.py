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
        self._encoder = dataset.get('_encoder', LabelEncoder())
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

        # Dynamically sets attributes based on dictionary keys.
        for key, value in dataset.items():
            # We check hasattr to ensure we only set attributes defined in __init__.  This prevents accidental
            # injection of arbitrary dictionary keys.
            if hasattr(self, key) and key != 'plant_species_cnt':
                setattr(self, key, value)
                
                # Fit the encoder once and only once when the species list is provided.
                # This ensures that 0, 1, 2... always map to the same species across all instances.
                if key == 'plant_species' and value:
                    self._encoder.fit(value)

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

    def _encode_label(self, y_data: pd.DataFrame) -> np.ndarray:
        # We rely on the encoder being pre-fitted in set_dataset to maintain consistent mapping between train, test,
        # and validation sets.
        return tf.keras.utils.to_categorical(
            self._encoder.transform(y_data),
            num_classes=self.plant_species_cnt
        )

    def normalize(self) -> None:
        self.x_train_norm = self.x_train.astype('float32') / IMAGE_PX_MAX
        self.x_test_norm = self.x_test.astype('float32') / IMAGE_PX_MAX
        self.x_val_norm = self.x_val.astype('float32') / IMAGE_PX_MAX

    def print_classification_report(self, model, x_data: np.ndarray, y_true_encoded: np.ndarray) -> None:
        y_true_labels = np.argmax(y_true_encoded, axis=1)
        y_pred_probs = model.predict(x_data)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print(classification_report(
            y_true_labels,
            y_pred_classes,
            target_names=self.plant_species,
            digits=4)
        )
