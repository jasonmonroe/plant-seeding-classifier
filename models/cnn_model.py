# models/cnn_model.py

import random
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

from models.modeler import Modeler

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

from src.utils import show_banner, early_stopping, reduce_lr, start_timer, show_timer

class CnnModel(Modeler):
    def __init__(self, dataset: dict):
        super().__init__(dataset=dataset)

        self.title = ''
        self.__init_session()
        self._reduce_lr = reduce_lr()
        self._early_stopping = early_stopping()

        self.optimizer = 'adam'
        self.model = Sequential()

        self.history = None
        self.loss = 0.0
        self.accuracy = 0.0
        self.y_test_pred = None
        self.y_train_pred = None
        self.training_perf = pd.DataFrame({})

    def __init_session(self) -> None:
        """
        Clears the current Keras session, resetting all layers and models previously created, freeing up memory and
        resources.
        """

        tf.keras.backend.clear_session()
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

    def show_summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

    def _get_class_weights(self):
        """
        Calculates balanced class weights based on the training labels.
        """
        # axis=1 ensures we get the max index for each sample, resulting in a 1D array
        y_train_classes = np.argmax(self.y_train_enc, axis=1)
        unique_classes = np.unique(y_train_classes)

        raw_class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train_classes
        )

        # Map unique class indices to their respective weights.
        return dict(zip(unique_classes, raw_class_weights))

    def fit_model(self) -> History:
        class_weights = self._get_class_weights()

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
            class_weight=class_weights,
            callbacks=[self._reduce_lr, self._early_stopping]
        )

        return self.history

    # Data Augmented, Transfer Learning
    def fit_trained_model(self, datagen, val_datagen=None) -> History:
        # Use np.ceil to ensure remainder/partial batches aren't dropped
        steps_per_epoch = int(np.ceil(self.x_train_norm.shape[0] / TRAINED_BATCH_SIZE))
        validation_steps = int(np.ceil(self.x_val_norm.shape[0] / TRAINED_BATCH_SIZE))
        class_weights = self._get_class_weights()

        # If no specific validation generator is provided, create a basic one
        if val_datagen is None:
            val_datagen = ImageDataGenerator()

        self.history = self.model.fit(
            datagen.flow(
                self.x_train_norm,
                self.y_train_enc,
                batch_size=TRAINED_BATCH_SIZE,
                seed=SEED,
                shuffle=True,
            ),
            epochs=TRAINED_EPOCH_CNT,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_datagen.flow(
                self.x_val_norm,
                self.y_val_enc,
                batch_size=TRAINED_BATCH_SIZE,
                shuffle=False
            ),
            validation_steps=validation_steps,
            verbose=1,
            class_weight=class_weights,
            callbacks=[self._reduce_lr, self._early_stopping]
        )

        return self.history

    def evaluate(self, verbose: int = 2) -> tuple[float, float]:
        print(f'\n{self.title} Evaluation')

        self.loss, self.accuracy = self.model.evaluate(
            self.x_test_norm,
            self.y_test_enc,
            verbose=verbose
        )

        print(f'Test Loss: {self.loss}, Test Accuracy: {self.accuracy}')

        return self.loss, self.accuracy

    def calc_performance(self) -> None:
        self.training_perf = self.get_model_performance_classification(self.model, self.x_train_norm, self.y_train_enc)

        print(f'\n--- {self.title} Training Performance ---')
        print(self.training_perf)

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.y_train_pred = self.model.predict(self.x_train_norm)
        self.y_test_pred = self.model.predict(self.x_test_norm)
        self.y_test_true_labels = np.argmax(self.y_test_enc, axis=1) # @todo - normalized?

        return self.y_train_pred, self.y_test_pred, self.y_test_true_labels

    def show_results(self) -> None:
        show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)
        show_banner(self.title, 'Classification Report')
        self.print_classification_report(self.model, self.x_test_norm, self.y_test_enc)

    def show_history(self) -> None:
        return None
        # Show plot history for accuracy, loss
        show_plot_history(self.history, self.title, 'accuracy')
        show_plot_history(self.history, self.title, 'loss')

    """
    ==================================
    MODEL PERFORMANCE CLASSIFICATION
    ==================================
    
    Metric, Working (Minimum Signal), Good (Expected Target), Excellent (Production Goal)
    Test Accuracy, 60%−75%, 75%−85%, >85%
    Test F1-Score, 0.55−0.70, 0.75−0.85, >0.85
    Test Loss, <1.5, <0.7, <0.4
    
    Define a function to compute different metrics to check performance of a classification model built using stats models
    """
    def get_model_performance_classification(self, model: Sequential, predictors: np.ndarray, target: np.ndarray) -> pd.DataFrame:
         
        """
        Function to compute different metrics to check classification model performance
        model: classifier
        predictors: independent variables
        target: target variable
        threshold: threshold for classification
        """

        # Use np.argmax to get the predicted class index
        # (i.e: predictors ~ x_training_normalized)
        predicted_labels = np.argmax(self.model.predict(predictors), axis=1)

        # Convert TRUE one-hot encoded target labels to integer class indices
        # (i.e: target ~ y_training_encoded)
        target_classes = np.argmax(target, axis=1)

        acc = accuracy_score(target_classes, predicted_labels)
        prec = precision_score(target_classes, predicted_labels, average='weighted')
        rec = recall_score(target_classes, predicted_labels, average='weighted')
        f1 = f1_score(target_classes, predicted_labels, average='weighted')

        df_perform = pd.DataFrame({
            'Accuracy': [acc],
            'Precision': [prec],
            'Recall': [rec],
            'F1': [f1]
        })

        return df_perform

    def run(self, datagen: ImageDataGenerator=None) -> None:
        print(f'\n--- Running cnn_model:{self.title} ---')
        """
        Run each model in a sequence:
        - Compile model
        - Show model summary
        - Fit the model with a data generator if necessary
        - Show the Plot History
        - Evaluate the model
        - Calculate the model performance
        - Get the predictions
        - Show all the results
        """
        #print(f'\n--- Running {self.title} ---')

        self.compile()
        self.show_summary()
        show_banner(self.title, '- Fitting Training Model -')

        start_time = start_timer()
        if datagen is None:
            self.fit_model()
        else:
            self.fit_trained_model(datagen)
        show_timer(start_time)

        self.show_history()
        self.evaluate()
        self.calc_performance()
        self.get_predictions()
        self.show_results()
