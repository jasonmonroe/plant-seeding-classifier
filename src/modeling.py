# modeler.py
# @todo - DELETE!
import random
from typing import Any, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D, # Import the GlobalAveragePooling2D layer
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator


from src.config import BASE_BATCH_SIZE, BASE_EPOCH_CNT, DROPOUT_RATE, IMAGE_ROWS, KERNEL_SIZE_MED, KERNEL_SIZE_SM, LG_CNT, MED_CNT, SEED, SM_CNT, TRAINED_BATCH_SIZE, TRAINED_EPOCH_CNT, XLG_CNT, XXLG_CNT


# Note: Can we set global variables in a script?
 

# Note: Assume label_encoder() is instantiated
def encode_label(data: pd.DataFrame, label_encoder: LabelEncoder, plant_species_cnt: int) -> np.ndarray:
    # Encode categorical features and scale the pixel values
    # Creating one-hot encoded representation of target labels

    return tf.keras.utils.to_categorical(
        label_encoder.fit_transform(data),
        num_classes=plant_species_cnt
    )

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# Used to one-hot encode integer labels into a binary matrix representation that's
# suitable for classification tasks in deep learing.
def encode_data(label_encoder, y_training_data: pd.DataFrame, y_testing_data: pd.DataFrame, y_validation_data: pd.DataFrame, plant_species_cnt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    y_training_encoded = encode_label(y_training_data, label_encoder, plant_species_cnt)
    y_testing_encoded = encode_label(y_testing_data, label_encoder, plant_species_cnt)
    y_validation_encoded = encode_label(y_validation_data, label_encoder, plant_species_cnt)

    return y_training_encoded, y_testing_encoded, y_validation_encoded

def get_model_predictions(model: Sequential, x_training_normalized: np.ndarray, x_testing_normalized: np.ndarray, y_testing_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_predictor_training = model.predict(x_training_normalized)
    y_predictor_testing = model.predict(x_testing_normalized)
    y_testing_normalized = np.argmax(y_testing_encoded, axis=1)

    return y_predictor_training, y_predictor_testing, y_testing_normalized

# Evaluate Model and return the accuracy and loss analysis
def evalute_model(model: Sequential, x_testing_normalized: np.ndarray, y_testing_encoded: np.ndarray) -> Tuple[float, float]:
    loss, accuracy = model.evaluate(
        x_testing_normalized,
        y_testing_encoded,
        verbose=2
    )

    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    return loss, accuracy

# Fit Model and return history.  Also time the optimization.
def fit_model(mod: Sequential, x_training_normalized: np.ndarray, y_training_encoded: np.ndarray, x_validation_normalized: np.ndarray, y_validation_encoded: np.ndarray) -> History:

    y_training_classes = np.argmax(y_training_encoded, axis=1)
    raw_class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_training_classes),
        y=y_training_classes
    )
    # Convert the array of weights to a dictionary mapping class indices to weights
    class_weights_dict = dict(enumerate(raw_class_weights))

    return mod.fit(
        x_training_normalized,
        y_training_encoded,
        epochs=BASE_EPOCH_CNT,
        validation_data=(
            x_validation_normalized,
            y_validation_encoded
        ),
        batch_size=BASE_BATCH_SIZE,
        verbose=2,
        class_weight=class_weights_dict,
        #callbacks=[reduce_lr, early_stopping]
    )

# Fit Model with trained generated data
def fit_trained_model(
        mod: Sequential,
        x_training_normalized: np.ndarray,
        y_training_encoded: np.ndarray,
        x_validation_normalized: np.ndarray,
        y_validation_encoded: np.ndarray,
        x_validation_data: np.ndarray,
        datagen: ImageDataGenerator,
        validation_generator: NumpyArrayIterator,
        reduce_lr: ReduceLROnPlateau,
        early_stopping: EarlyStopping
) -> History:

    return mod.fit(
        datagen.flow(
            x_training_normalized,
            y_training_encoded,
            batch_size=TRAINED_BATCH_SIZE,
            seed=SEED,
            shuffle=True,
        ),
        epochs=TRAINED_EPOCH_CNT,
        steps_per_epoch=(x_training_normalized.shape[0] // TRAINED_BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=(x_validation_data.shape[0] // TRAINED_BATCH_SIZE),
        verbose=1,
        callbacks=[reduce_lr, early_stopping]
    )







# Visualizing the predicted and correct label of images from test data, add `pixel height`, `pixel width` labels
"""
def show_visualize_prediction(
        mod: Sequential,
        enc: LabelEncoder,
        x_test: np.ndarray,
        x_testing_norm: np.ndarray,
        y_testing_enc: np.ndarray,
        image_height,
        image_width,
        image_channels,
        show_all: bool=False,
    ) -> Tuple[int, int]:

        correct_cnt = 0
        image_cnt = len(x_testing_norm)
        show_cnt = IMAGE_ROWS

        if show_all:
            show_cnt = image_cnt

        for i in range(0, show_cnt):
            if show_all:
                index = i
            else:
                index = random.randint(0, image_cnt)

            print(f'# ----- [{i} of {show_cnt}] ----- #')
            print(f'Index:{index}\n')

            # Get the predicted probabilities
            predicted_probs = mod.predict((x_testing_norm[index].reshape(1, image_height, image_width, image_channels)), verbose=0)

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predicted_probs, axis=1)

            # Use the index to get the predicted label
            predicted_label = enc.inverse_transform(predicted_class_index)
            predicted_label = predicted_label[0]

            # Using inverse_transform() to get the output label from the output vector.
            true_label_index = np.argmax(y_testing_enc[index])  # Get index of true label
            true_label = enc.classes_[true_label_index]         # Get true label from index

            plt.figure(figsize=(2, 2))
            plt.imshow(x_test[index])
            plt.show()

            print('Predicted Label:', predicted_label)
            print('True Label: ', true_label)

            if predicted_label == true_label:
                print('✅ Correct Prediction!')
                correct_cnt += 1
            else:
                print('❌ Incorrect Prediction')

            print('# ----- [END] ----- #\n\n')

        return correct_cnt, show_cnt
"""




# --- Model Creation 

# Note: This can be a child class of Modeling
"""
def create_base_model(plant_species_cnt: int, image_params: tuple):
    return Sequential([
        # --- Convolution Block 1 ---
        Conv2D(SM_CNT, KERNEL_SIZE_MED, activation='relu', padding='same', input_shape=image_params),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        # --- Convolution Block 2 ---
        Conv2D(MED_CNT, KERNEL_SIZE_MED, activation='relu', padding='same'),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        # --- Convolution Block 3 ---
        Conv2D(LG_CNT, KERNEL_SIZE_MED, activation='relu', padding='same'),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        Conv2D(SM_CNT, KERNEL_SIZE_MED, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        # --- Classifier ---
        Flatten(),
        Dense(LG_CNT, activation='relu'),
        Dropout(DROPOUT_RATE),               # helps prevent overfitting on small datasets
        Dense(plant_species_cnt, activation='softmax')
    ])


def create_data_augmented_model(plant_species_cnt: int, image_params: tuple):
    return Sequential([

        # Block 1
        Conv2D(SM_CNT, KERNEL_SIZE_MED, padding='same', input_shape=image_params),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        # Block 2
        Conv2D(MED_CNT, KERNEL_SIZE_MED, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        # Block 3
        Conv2D(LG_CNT, KERNEL_SIZE_MED, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=KERNEL_SIZE_SM),

        # Glabal Pooling and Dense Layers
        GlobalAveragePooling2D(),
        Dense(XLG_CNT),
        BatchNormalization(),
        Activation('relu'),
        Dropout(DROPOUT_RATE),
        Dense(plant_species_cnt, activation='softmax'),
    ])


def create_vgg_model(image_params):
    return VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=image_params
    )


def create_transfer_learning_model(vgg_model, plant_species_cnt: int):
    return Sequential([
        vgg_model,
        GlobalAveragePooling2D(),  # converts 2D features to 1D vector
        Dense(XXLG_CNT, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(plant_species_cnt, activation='softmax')
    ])


def get_accuracy_data():
    pass


def show_accuracy():
    pass
"""