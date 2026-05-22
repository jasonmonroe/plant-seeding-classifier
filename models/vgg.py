# vgg.py

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg16 import VGG16

from models.cnn_model import CnnModel


class Vgg(CnnModel):
    def __init__(self, params):
        self.title = 'VGG26 Model'
        self.image_params = params
        self.head_output = self.get_head_output()


    def _create(self):
        self.model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_params
        )

    def get_head_input(self):
        return GlobalAveragePooling2D()(self.model.output)

    def get_head_output(self):
        return Dense(units=self.plant_species_cnt, activation='softmax')(self.get_head_input())

