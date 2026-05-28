# models/cnn_model.py

import random
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History

from models.modeler import Modeler
from notebooks.plant_seed_classification import show_banner
from src.config import (
    BASE_BATCH_SIZE, 
    BASE_EPOCH_CNT, 
    SEED, 
    TRAINED_BATCH_SIZE, 
    TRAINED_EPOCH_CNT
)
from src.eda import (
    show_plot_confusion_matrix, 
    show_plot_history
)


class CnnModel(Modeler):
    def __init__(self, plant_species, x_train, y_train, x_val, y_val, x_test, y_test):
        super().__init__(plant_species, x_train, y_train, x_val, y_val, x_test, y_test)
        self.__init_session()
        self._reduce_lr = self.reduce_lr()
        self._early_stopping = self.early_stopping()
        self.optimizer = None

        self.title = ''
        self.model = None

        self.history = None
        self.loss = None
        self.accuracy = None

        self.y_test_pred = None
        self.y_train_pred = None
        
        self.training_perf = None

    def __init_session(self) -> None:
        """
        Clears the current Keras session, resetting all layers and models previously created, 
        freeing up memory and resources.
        """

        print('__init_session()')
        tf.keras.backend.clear_session()
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

    def show_summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self):
        y_train_classes = np.argmax(self.y_train_enc)

        raw_class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_classes),
            y=y_train_classes
        )

        # Convert the array of weights to a dictionary mapping class indices to weights
        class_weights_dict = dict(enumerate(raw_class_weights))

        # Get History
        self.history = self.model.fit(
            self.x_train_norm,
            self.y_train_enc,
            epochs=BASE_EPOCH_CNT,
            validation_data=(
                self.x_val_norm,
                self.y_val_enc
            ),
            batch_size=BASE_BATCH_SIZE,
            verbose=2,
            class_weight=class_weights_dict,
            callbacks=[self._reduce_lr, self._early_stopping]
        )

    # Data Augmented, Transfer Learning
    def fit_trained_model(self, datagen, val_generator) -> History:
        self.history = self.model.fit(
            datagen.flow(
                self.x_train_norm,
                self.y_train_enc,
                batch_size=TRAINED_BATCH_SIZE,
                seed=SEED,
                shuffle=True,
            ),
            epochs=TRAINED_EPOCH_CNT,
            steps_per_epoch=(self.x_train_norm.shape[0] // TRAINED_BATCH_SIZE),
            validation_data=val_generator,
            validation_steps=(self.x_val.shape[0] // TRAINED_BATCH_SIZE),
            verbose=1,
            callbacks=[self._reduce_lr, self._early_stopping]
        )

        return self.history

    def evaluate(self)-> tuple[Any | None, Any]:
        
        show_banner(self.title, 'Evaluation')

        self.loss, self.accuracy = self.model.evaluate(
            self.x_test_norm,
            self.y_test_enc,
            verbose=2 # used for base but not for data_augmented model
        )

        print(f'Test Loss: {self.loss}, Test Accuracy: {self.accuracy}')

        return self.loss, self.accuracy

    def calc_performance(self):
        self.training_perf = show_model_performance_classification(self.model, self.x_train_norm, self.y_train_enc)

        print('Training Performance')
        print(self.training_perf)

        return self.training_perf

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.y_train_pred = self.model.predict(self.x_train_norm)
        self.y_test_pred = self.model.predict(self.x_test_norm)
        self.y_test_norm = np.argmax(self.y_test_enc, axis=1)
        
        return self.y_train_pred, self.y_test_pred, self.y_test_norm

    def show_results(self):
        show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)
        show_banner(self.title, 'Classification Report')
        print_classification_report(self.model, self.x_test_norm, self.y_test_enc, self.plant_species)

    def show_history(self):
        # Show plot history for accuracy, loss
        show_plot_history(self.history, self.title, 'accuracy')
        show_plot_history(self.history, self.title, 'loss')

    def run(self):
        # compile, show summary, show banner, fit, show plot history, evaluate, calc perf, get predictions,
        # show results,
        pass
