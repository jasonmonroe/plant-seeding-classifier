# transfer_learning.py

import random
from typing import Tuple, Optional, List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input, layers
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

        # Safe, unified optimization reference
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)

        # Extract parameters from the wrapper object
        self.image_params = vgg16_model.image_params
        vgg_base = vgg16_model.get_base()
        self._create(vgg_base)

    def _create(self, vgg16_base_model) -> None:
        print(f'\n* Creating {self.title} *')
        
        # 1. Define Input
        inputs = Input(shape=self.image_params)

        x = layers.Lambda(lambda img: tf.keras.applications.vgg16.preprocess_input(
            tf.where(
                tf.reduce_max(img) <= 1.0,
                img * 255.0,
                img
            )
        ), name='vgg16_preprocess')(inputs)

        x = vgg16_base_model(x, training=False)

        # 4. Add Custom Head
        x = GlobalAveragePooling2D()(x)
        x = Dense(XXLG_CNT, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        outputs = Dense(self.plant_species_cnt, activation='softmax')(x)
        
        # 5. Build Functional Model
        self.model = Model(inputs, outputs, name="vgg16_refined")

    # Visualizing the predicted and correct label of images from test data, add `pixel height`, `pixel width` labels



    def show_prediction_visualization2(
            self,
            image_params: Tuple[int, int, int],
            show_all: bool = False,
            target_indices: Optional[List[int]] = None
    ) -> Tuple[int, int]:

        image_width, image_height, image_channels = image_params
        correct_cnt = 0
        image_cnt = len(self.x_test_norm)

        # Calculate full matrix predictions deterministically
        print('\n* Generating Global Evaluation Slice Predictions... *')
        all_predicted_probs = self.model.predict(self.x_test_norm, batch_size=32, verbose=0)
        all_predicted_indices = np.argmax(all_predicted_probs, axis=1)
        all_true_indices = np.argmax(self.y_test_enc, axis=1)

        # Determine indices to visualize
        if target_indices is not None:
            indices_to_run = target_indices
            show_cnt = len(indices_to_run)
        elif show_all:
            indices_to_run = list(range(image_cnt))
            show_cnt = image_cnt
        else:
            show_cnt = IMAGE_ROWS
            indices_to_run = random.sample(range(image_cnt), show_cnt)

        for i, index in enumerate(indices_to_run):
            print('# --- [START] --- #')
            print(f'--- [{i + 1} of {show_cnt}] ---')
            print(f'Dataset Random Index: {index}')

            predicted_class_index = all_predicted_indices[index]
            true_label_index = all_true_indices[index]

            print(f'predicted_class_index: {predicted_class_index}, true_label_index: {true_label_index}')

            # Map index names explicitly using global state
            predicted_label = self._encoder.inverse_transform([predicted_class_index])[0]
            true_label = self._encoder.inverse_transform([true_label_index])[0]

            plt.figure(num=f"Predictions (Index {index})", figsize=(4, 4))

            # Incorporate explicit dimension markers requested in your layout comments
            plt.xlabel(f"Width: {image_width}px")
            plt.ylabel(f"Height: {image_height}px")

            try:
                if hasattr(self, 'x_test') and self.x_test is not None:
                    plt.imshow(self.x_test[index])
                else:
                    plt.imshow(np.squeeze(self.x_test_norm[index]))
            except Exception:
                plt.imshow(np.squeeze(self.x_test_norm[index]))

            plt.title(f"True: {true_label}\nPred: {predicted_label}")
            plt.tight_layout()
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


        def show_prediction_visualization(
                self,
                image_params: Tuple[int, int, int],
                show_all: bool = False,
                target_indices: Optional[List[int]] = None
        ) -> Tuple[int, int]:

            image_width, image_height, image_channels = image_params

            correct_cnt = 0
            image_cnt = len(self.x_test_norm)
            print('\n* Predicting images... *')
            print(f'x_test_norm count: {len(self.x_test_norm)}, y_test_enc count:{len(self.y_test_enc)}, x_test count: {len(self.x_test)}')

            # ====================================================================
            # CRITICAL DEBUG 1: EXPOSE GLOBAL ENCODER MAPPING VS DATA LABELS
            # ====================================================================
            print("\n=== DEBUG: GLOBAL ENCODER STATE ===")
            if hasattr(self._encoder, 'classes_'):
                print("LabelEncoder Classes:", list(self._encoder.classes_))
                for idx, name in enumerate(self._encoder.classes_):
                    print(f"  Encoder Index {idx} -> Name: '{name}'")
            else:
                print("❌ WARNING: self._encoder has not been fitted yet!")

            # Check the first few raw one-hot encoded rows in y_test_enc to see if they make sense
            print("\n=== DEBUG: TEST GROUND TRUTH ENCODING ===")
            for test_i in range(min(3, image_cnt)):
                raw_one_hot = self.y_test_enc[test_i]
                detected_true_idx = np.argmax(raw_one_hot)
                print(f"  Sample {test_i} one-hot vector: {raw_one_hot} -> Argmax Index: {detected_true_idx}")
            print("===========================================\n")
            # ====================================================================

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
                print('# --- [START] --- #')
                print(f'--- [{i + 1} of {show_cnt}] ---')
                print(f'Dataset Random Index: {index}')

                # Get the predicted probabilities
                test_image = self.x_test_norm[index].reshape(-1, image_height, image_width, image_channels)
                predicted_probs = self.model.predict(test_image, verbose=0)

                # ====================================================================
                # CRITICAL DEBUG 2: EXPOSE RAW MODEL PROBABILITIES OUT FOR SAMPLE
                # ====================================================================
                print(f"  Raw Model Output Probabilities:\n  {predicted_probs[0]}")
                # ====================================================================

                # Get the index of the class with the highest probability
                predicted_class_index = int(np.argmax(predicted_probs, axis=1)[0])
                true_label_index = int(np.argmax(self.y_test_enc[index]))
                print(f'predicted_class_index: {predicted_class_index}, true_label_index: {true_label_index}')

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

        def show_results(self, show_all) -> None:
            """
            Orchestrates the visualization of model performance.
            """
            # @todo - show_plot_confusion_matrix(self.y_test_enc, self.y_test_pred)



            prediction_correct, total = self.show_prediction_visualization(self.image_params, show_all)
            pct = (prediction_correct / total) * 100
            show_banner(f'{self.title} Results', f'{prediction_correct} / {total} Correct\nAccuracy: {pct:.2f}%')
