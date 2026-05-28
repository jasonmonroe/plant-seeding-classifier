# models/modeler.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from sklearn.metrics import (
    classification_report
)

from src.config import IMAGE_PX_MAX

# Parent Class
# 1. Encode data, then normalize it
class Modeler:
    def __init__(self, plant_species, dataset: dict):
        self._encoder = LabelEncoder()
        self.plant_species = plant_species
        self.plant_species_cnt = len(self.plant_species)

        # Data variables
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        # Encoded variables
        self.y_train_enc = None
        self.y_val_enc = None
        self.y_test_enc = None

        # Normalized variables
        self.x_train_norm = None
        self.x_val_norm = None
        self.x_test_norm = None
        self.y_test_norm = None

        # Set dataset values
        self.__dict__.update(dataset)
        #self.set_dataset(dataset)

    def set_dataset(self, dataset: dict):
        print('set_dataset()')
        for key, value in dataset.items():
            setattr(self, key, value)
            print(f'Debug: set {key} = {value}')

    def get_proc_dataset(self):
        """
        Get processed data that has been encoded and normalized.
        :return: dict of processed data.
        """
        return {
            "x_train_norm": self.x_train_norm,
            "x_test_norm": self.x_test_norm,
            "x_val_norm": self.x_val_norm,
            "y_train_enc": self.y_train_enc,
            "y_test_enc": self.y_test_enc,
            "y_val_enc": self.y_val_enc,
        }

    def encode_data(self):
        self.y_train_enc = self._encode_label(self.y_train)
        self.y_test_enc = self._encode_label(self.y_test)
        self.y_val_enc = self._encode_label(self.y_val)
        #return y_training_encoded, y_testing_encoded, y_validation_encoded

    def _encode_label(self, y_data: pd.DataFrame):
        return tf.keras.utils.to_categorical(
            self._encoder.fit_transform(y_data),
            num_classes=self.plant_species_cnt
        )

    def normalize(self):
        self.x_train_norm = self.x_train.astype('float32') / IMAGE_PX_MAX
        self.x_test_norm = self.x_test.astype('float32') / IMAGE_PX_MAX
        self.x_val_norm = self.x_val.astype('float32') / IMAGE_PX_MAX

    def print_classification_report(self, model, x_data: np.ndarray, y_true_encoded: np.ndarray):
        y_true_labels = np.argmax(y_true_encoded, axis=1)
        y_pred_probs = model.predict(x_data)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print(classification_report(
            y_true_labels,
            y_pred_classes,
            target_names=self.plant_species,
            digits=4)
        )
