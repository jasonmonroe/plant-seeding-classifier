# transfer_learning.py

import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
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

"""
Transfer Learning Model

The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task to 
solve a specific, smaller task.
"""

class TransferLayerModel(CnnModel):
    def __init__(self, vgg_model):
        self.title = 'Transfer Learning Model'
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)
        self._create(vgg_model)

    def _create(self, vgg_model):
        self.model = Sequential([
        vgg_model,
        GlobalAveragePooling2D(),  # converts 2D features to 1D vector
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
                index = random.randint(0, image_cnt)

            print(f'# ----- [{i} of {show_cnt}] ----- #')
            print(f'Index: {index}\n')

            # Get the predicted probabilities
            predicted_probs = self.model.predict((self.x_test_norm[index].reshape(1, image_height, image_width, image_channels)), verbose=0)

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predicted_probs, axis=1)

            # Use the index to get the predicted label
            predicted_label = self.encoder.inverse_transform(predicted_class_index)
            predicted_label = predicted_label[0]

            # Using inverse_transform() to get the output label from the output vector.
            true_label_index = np.argmax(self.y_test_enc[index])  # Get index of true label
            true_label = self.encoder.classes_[true_label_index]  # Get true label from index

            plt.figure(figsize=(2, 2))
            plt.imshow(self.x_test[index])
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

    # Override parent class show_results
    def show_results(self, image_params):
        show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)
        
        prediction_correct, total = self.show_visualize_prediction(
            #self.x_test,
            #self.x_test_norm,
            #self.y_test_enc,
            image_params
            #image_handle.height,
            #image_handle.width,
            #image_handle.channels
        )

        #show_timer(start_time)

        pct = (prediction_correct / total) * 100
        show_banner(self.title, f'{prediction_correct} / {total} \nPrediction Accuracy: {pct:.2f}%')
