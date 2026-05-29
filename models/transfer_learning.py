# transfer_learning.py

import random
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from models.cnn_model import CnnModel
from src.config import DROPOUT_RATE, IMAGE_ROWS, TL_LEARNING_RATE, XXLG_CNT
from src.eda import show_plot_confusion_matrix
from src.utils import show_banner


class TransferLayerModel(CnnModel):
    """
    Transfer Learning Model

    The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task
    to solve a specific, smaller task.
    """

    def __init__(self, vgg16_model, dataset: dict):
        super().__init__(dataset=dataset)

        self.title = 'Transfer Learning Model'
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)
        
        # Extract parameters from the wrapper object
        self.image_params = vgg16_model.image_params
        vgg_base = vgg16_model.get_base()
        self._create(vgg_base)

    def _create(self, vgg16_base_model) -> None:
        print(f'\n* Creating {self.title} *')
        
        # We must "unwrap" the Keras model from the VggModel wrapper.
        # vgg16_model.model is the actual Keras Functional model.
        self.model = Sequential([
            vgg16_base_model,
            GlobalAveragePooling2D(),  
            Dense(XXLG_CNT, activation='relu'),
            Dropout(DROPOUT_RATE),
            Dense(self.plant_species_cnt, activation='softmax')
        ])

    # Visualizing the predicted and correct label of images from test data, add `pixel height`, `pixel width` labels
    def show_visualize_prediction(
        self,
        image_params: Tuple[int, int, int],
        show_all: bool = False,
        target_indices: Optional[List[int]] = None
    ) -> Tuple[int, int]:

        image_width, image_height, image_channels = image_params

        correct_cnt = 0
        image_cnt = len(self.x_test_norm)

        # Determine which indices to evaluate
        if target_indices is not None:
            indices_to_run = target_indices
            show_cnt = len(indices_to_run)
        elif show_all:
            indices_to_run = list(range(image_cnt))
            show_cnt = image_cnt
        else:
            show_cnt = IMAGE_ROWS
            # Guarantee unique random selections
            indices_to_run = random.sample(range(image_cnt), show_cnt)

        for i, index in enumerate(indices_to_run):
            print(f'# --- [{i + 1} of {show_cnt}] --- #')
            print(f'Dataset Index: {index}')

            # Get the predicted probabilities
            # Using -1 for the batch dimension is a more robust way to reshape for single predictions
            test_image = self.x_test_norm[index].reshape(-1, image_height, image_width, image_channels)
            predicted_probs = self.model.predict(test_image, verbose=0)

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predicted_probs, axis=1)[0]
            true_label_index = np.argmax(self.y_test_enc[index])

            # Decode strings using identical transformer logic
            predicted_label = self._encoder.inverse_transform([predicted_class_index])[0]
            true_label = self._encoder.inverse_transform([true_label_index])[0]

            plt.figure(num=f"Predictions (Index {index})", figsize=(3, 3))
            try:
                if hasattr(self, 'x_test') and self.x_test is not None:
                    plt.imshow(self.x_test[index])
                else:
                    plt.imshow(np.squeeze(self.x_test_norm[index]))
            except Exception:
                plt.imshow(np.squeeze(self.x_test_norm[index]))

            plt.axis('off')
            plt.show()

            print('Predicted Label:', predicted_label)
            print('True Label:     ', true_label)

            if predicted_class_index == true_label_index:
                print('✅ Correct Prediction!')
                correct_cnt += 1
            else:
                print('❌ Incorrect Prediction')
            print('# --- [END] --- #\n')

        return correct_cnt, show_cnt

    def show_results(self) -> None:
        """
        Orchestrates the visualization of model performance.
        """
        show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)
        
        prediction_correct, total = self.show_visualize_prediction(self.image_params)
        pct = (prediction_correct / total) * 100
        show_banner(self.title, f'{prediction_correct} / {total} Correct\nAccuracy: {pct:.2f}%')
