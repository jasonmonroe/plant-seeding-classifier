# vgg.py

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg16 import VGG16

from models.cnn_model import CnnModel
from src.config import LG_CNT, IMAGE_CHANNELS

# @link https://keras.io/api/applications/vgg/vgg_models/
class VggModel(CnnModel):
    def __init__(self, dataset: dict):
        super().__init__(dataset=dataset)

        self.title = 'VGG16 Model'
        self.image_params = (LG_CNT, LG_CNT, IMAGE_CHANNELS)

        # Initialize the base VGG16 model
        self._base = self._create_base_model()

        # Build the final model by connecting the VGG base to the custom head
        self._create(self._get_head_output())

    def _create_base_model(self):
        print(f'* Creating {self.title} *')
        return VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_params,
        )

    def _create(self, head_output: Dense) -> Model:
        self.model = Model(
            inputs=self._base.input,
            outputs=head_output,
            name='vgg16_classifier'
        )

    def _get_head_input(self):
        # Takes the 4D output tensor from VGG16 and flattens it via pooling
        return GlobalAveragePooling2D()(self._base.output)

    def _get_head_output(self) -> Dense:
        return Dense(
            units=self.plant_species_cnt,
            activation='softmax'
            )(self._get_head_input())

    def get_base(self) -> VGG16:
        return self._base
