# data_augm.py

from keras.src.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)

from models.cnn_model import CnnModel
from src.config import (
    KERNEL_SIZE_MED, 
    SM_CNT, 
    KERNEL_SIZE_SM, 
    MED_CNT, 
    LG_CNT, 
    XLG_CNT, 
    DROPOUT_RATE, 
    DA_LEARNING_RATE
)


class DataAugmentedModel(CnnModel):
    def __init__(self, params, dataset):
        super().__init__(dataset=dataset)
        self.title = 'Data Augmented CNN Model'
        self.image_params = params
        self.optimizer = Adam(learning_rate=DA_LEARNING_RATE)

        self._load_dataset(dataset)
        self._create()

    def _create(self):
         self.model = Sequential([
             Conv2D(SM_CNT, KERNEL_SIZE_MED, padding='same', input_shape=self.image_params),
             BatchNormalization(),
             Activation('relu'),
             MaxPooling2D(pool_size=KERNEL_SIZE_SM),

             # --- Block 2 ---
             Conv2D(MED_CNT, KERNEL_SIZE_MED, padding='same'),
             BatchNormalization(),
             Activation('relu'),
             MaxPooling2D(pool_size=KERNEL_SIZE_SM),

             # --- Block 3 ---
             Conv2D(LG_CNT, KERNEL_SIZE_MED, padding='same'),
             BatchNormalization(),
             Activation('relu'),
             MaxPooling2D(pool_size=KERNEL_SIZE_SM),

             # --- Global Pooling and Dense Layers ---
             GlobalAveragePooling2D(),
             Dense(XLG_CNT),
             BatchNormalization(),
             Activation('relu'),
             Dropout(DROPOUT_RATE),
             Dense(self.plant_species_cnt, activation='softmax')
         ])
