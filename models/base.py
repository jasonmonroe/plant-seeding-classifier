# models/base.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

from models.cnn_model import CnnModel
from src.config import DROPOUT_RATE, KERNEL_SIZE_MED, KERNEL_SIZE_SM, LG_CNT, MED_CNT, SM_CNT


class BaseModel(CnnModel):
    def __init__(self, image_params, dataset):
        super().__init__(dataset=dataset)

        self.title = 'CNN Base Model'
        self.image_params = image_params
        self.optimizer = 'adam'
        self._create()

    def _create(self) -> None:
        print(f'\n* Creating {self.title} *')
        self.model = Sequential([
            # --- Convolution Block 1 ---
            Conv2D(SM_CNT, KERNEL_SIZE_MED, activation='relu', padding='same', input_shape=self.image_params),
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
            Dropout(DROPOUT_RATE), # helps prevent overfitting on small datasets
            Dense(self.plant_species_cnt, activation='softmax')
        ])
    