# transfer_learning.py

import random
from typing import Tuple
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

    The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task to
    solve a specific, smaller task.
    """

    def __init__(self, vgg16_model, dataset: dict):
        super().__init__(dataset=dataset)

        self.title = 'Transfer Learning Model'
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)
        
        # Extract parameters from the wrapper object
        self.image_params = vgg16_model.image_params
        vgg_base = vgg16_model.get_base()
        self._create(vgg_base)

    def _create(self, vgg16_base_model):
        print(f'* Creating {self.title} *')
        
        # We must "unwrap" the Keras model from the VggModel wrapper.
        # vgg16_model.model is the actual Keras Functional model.
        self.model = Sequential([
            vgg16_base_model,
            # Converts 2D features to 1D vector. 
            # Note: Ensure VggModel.model output is 4D (include_top=False and no pooling).
            GlobalAveragePooling2D(),  
            Dense(XXLG_CNT, activation='relu'),
            Dropout(DROPOUT_RATE),
            Dense(self.plant_species_cnt, activation='softmax')
        ])

    # Visualizing the predicted and correct label of images from test data, add `pixel height`, `pixel width` labels
    def show_visualize_prediction(
        self,
        image_params,
        show_all: bool=False,
    ) -> Tuple[int, int]:

        image_width = image_params[0]
        image_height = image_params[1] 
        image_channels = image_params[2]

        correct_cnt = 0
        image_cnt = len(self.x_test_norm)
        show_cnt = IMAGE_ROWS

        if show_all:
            show_cnt = image_cnt

        for i in range(0, show_cnt):
            if show_all:
                index = i
            else:
                # randrange is exclusive of the stop value, preventing "out of range" errors
                index = random.randrange(image_cnt)

            print(f'# --- [{i} of {show_cnt}] --- #')
            print(f'Index: {index}\n')

            # Get the predicted probabilities
            # Using -1 for the batch dimension is a more robust way to reshape for single predictions
            test_image = self.x_test_norm[index].reshape(-1, image_height, image_width, image_channels)
            predicted_probs = self.model.predict(test_image, verbose=0)

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predicted_probs, axis=1)

            # Use the index to get the predicted label
            predicted_label = self._encoder.inverse_transform(predicted_class_index)
            predicted_label = predicted_label[0]

            # Extract the true index from the one-hot encoded vector
            true_label_index = np.argmax(self.y_test_enc[index])
            # Use inverse_transform to ensure consistency with the predicted label logic
            true_label = self._encoder.inverse_transform([true_label_index])[0]

            plt.figure(num=f"{self.title} Predictions ({index})", figsize=(4, 4))

            try:
                plt.imshow(self.x_test[index])
            except IndexError:
                # Fallback: Squeeze out any extra dimensions and plot the normalized array directly
                plt.imshow(np.squeeze(self.x_test_norm[index]))

            plt.show()

            print('Predicted Label:', predicted_label)
            print('True Label: ', true_label)

            if predicted_label == true_label:
                print('✅ Correct Prediction!')
                correct_cnt += 1
            else:
                print('❌ Incorrect Prediction')

            print('# --- [END] --- #\n\n')

        return correct_cnt, show_cnt

    # Override parent class show_results
    def show_results(self):
        show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)
        prediction_correct, total = self.show_visualize_prediction(self.image_params)

        pct = (prediction_correct / total) * 100
        show_banner(self.title, f'{prediction_correct} / {total} \nPrediction Accuracy: {pct:.2f}%')
