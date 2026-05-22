# transfer_learning.py
from models.cnn_model import CnnModel

from keras.src.optimizers import Adam
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

from src.config import DROPOUT_RATE, TL_LEARNING_RATE, XXLG_CNT

"""
Transfer Learning Model

The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task to 
solve a specific, smaller task.
"""

class TransferLayerModel(CnnModel):
    def __init__(self, vgg_model):
        self.title = 'Transfer Learning Model'
        self._create(vgg_model)

    def _create(self, vgg_model):
        self.model = Sequential([
        vgg_model,
        GlobalAveragePooling2D(),  # converts 2D features to 1D vector
        Dense(XXLG_CNT, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(self.plant_species_cnt, activation='softmax')
    ])

    def compile(self):
        self.model.compile(
            optimizer=Adam(learning_rate=TL_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def show_prediction():
        pass
