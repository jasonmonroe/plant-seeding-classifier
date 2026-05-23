# models/modeler.py
import numpy as np

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src.config import IMAGE_PX_MAX, IMAGE_ROWS, L2_LEARNING_RATE

# Parent Class
# 1. Encode data, then normalize it
class Modeler:
    def __init__(self, plant_species, x_train, y_train, x_val, y_val, x_test, y_test):
        self.encoder = LabelEncoder()
        self.plant_species = plant_species
        self.plant_species_cnt = len(self.plant_species)

        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

        self.x_test = x_test
        self.y_test = y_test

        self.y_train_enc = None
        self.y_val_enc = None
        self.y_test_enc = None

        self.x_train_norm = None
        self.x_val_norm = None
        self.x_test_norm = None
        self.y_test_norm = None

    def encode_data(self):
        self.y_train_enc = self._encode_label(self.y_train)
        self.y_test_enc = self._encode_label(self.y_test)
        self.y_val_enc = self._encode_label(self.y_val)

        #return y_training_encoded, y_testing_encoded, y_validation_encoded

    def _encode_label(self, data):
        return tf.keras.utils.to_categorical(
            self.encoder.fit_transform(data),
            num_classes=self.plant_species_cnt
        )

    def normalize(self):
        self.x_train_norm = self.x_train.astype('float32') / IMAGE_PX_MAX
        self.x_test_norm = self.x_test.astype('float32') / IMAGE_PX_MAX
        self.x_val_norm = self.x_val.astype('float32') / IMAGE_PX_MAX


    def reduce_lr(self):
        """
        Model Performance Improvement

        Reducing the Learning Rate:

        Hint:
        Use ReduceLRonPlateau() function that will be used to decrease the learning rate by some factor, if the loss is
        not decreasing for some time. This may start decreasing the loss at a smaller learning rate. There is a
        possibility that the loss may still not decrease. This may lead to executing the learning rate reduction again
        in an attempt to achieve a lower loss.
        """
        return ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=L2_LEARNING_RATE,
            verbose=1
        )

    def early_stopping(self):
        """
        Early Stopping

        Monitor validation loss and stop training automatically when the loss starts consistently increasing as a sign of
        overfitting.
        """

        return EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

    def print_classification_report(mod, x_data: np.ndarray, y_true_encoded: np.ndarray, plant_species: list):
        y_true_labels = np.argmax(y_true_encoded, axis=1)
        y_pred_probs = mod.predict(x_data)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print(classification_report(
            y_true_labels,
            y_pred_classes,
            target_names=plant_species,
            digits=4)
        )


   


    def build_metrics(self):
        pass

    def get_accuracy(self):
        pass


    def create_matrix(self):
        pass