# vgg.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg16 import VGG16

from models.cnn_model import CnnModel
from src.config import LG_CNT, IMAGE_CHANNELS

# @link https://keras.io/api/applications/vgg/vgg_models
class VggModel(CnnModel):
    def __init__(self, image_params, dataset: dict):
        super().__init__(dataset=dataset)

        self.title = 'VGG16 Model'
        self.image_params = image_params

        # Initialize the base VGG16 model and lock pre-trained weights by default
        self._base = self._create_base_model()

        self._base.trainable = True # False
        for layer in self._base.layers[:-4]:
            layer.trainable = False

        # Construct the final Functional Model
        self._create()

    def _create_base_model(self) -> VGG16:
        print(f'\n--- Creating {self.title} ---')
        return VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_params,
        )

    def _create(self) -> None:
        # Connect the 4D output of VGG to the 2D classification head
        head_input = GlobalAveragePooling2D()(self._base.output)
        head_output = Dense(
            units=self.plant_species_cnt,
            activation='softmax',
            name='species_projection'
        )(head_input)

        self.model = Model(
            inputs=self._base.input,
            outputs=head_output,
            name='vgg16_classifier'
        )

    def get_base(self) -> VGG16:
        return self._base