# transfer_learning.py

import random
from typing import Tuple, Optional, List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import Model
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input

from models.vgg import VggModel
from models.cnn_model import CnnModel
from src.config import (
    DROPOUT_RATE,
    IMAGE_NORMALIZED,
    IMAGE_PX_MAX,
    IMAGE_ROWS,
    TL_LEARNING_RATE,
    TRAINED_BATCH_SIZE,
    XXLG_CNT
)
from src.eda import show_plot_confusion_matrix
from src.utils import show_banner


class TransferLayerModel(CnnModel):
    """
    Transfer Learning Model

    Leverages pre-trained ImageNet feature weights from a frozen VGG16 base
    to accurately classify target plant seedling images.
    """

    def __init__(self, vgg16_model: VggModel, dataset: dict, eda: bool = False, all_pred: bool = False):
        super().__init__(dataset=dataset)

        self.title = 'Transfer Learning Model'
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)

        # Map initialization settings to class properties
        self.image_params = vgg16_model.image_params
        self.eda = eda
        self.all_pred = all_pred

        self._create(vgg16_model.get_base())

    def _create(self, vgg16_base_model: Model) -> None:
        print(f'\n--- Creating {self.title} ---')

        inputs = Input(shape=self.image_params)
        print(f'inputs={type(inputs)}')
        outputs = self.__build_outputs(vgg16_base_model, inputs)

        # Construct unified functional graph instance
        self.model = Model(inputs, outputs, name="vgg16_refined")

    def __build_outputs(self, vgg16_base_model: Model, inputs: tf.Tensor) -> Dense:
        print('__build_output()')

        # Hardened scaling normalization layer
        tensor_img = layers.Lambda(lambda img: preprocess_input(
            tf.where(
                tf.reduce_max(img) <= IMAGE_NORMALIZED,
                img * IMAGE_PX_MAX,
                img
            )
        ), name='vgg16_preprocess')(inputs)

        # Feature processing via frozen base layer
        tensor_img = vgg16_base_model(tensor_img, training=False)

        # Custom Dense classification head connection
        tensor_img = GlobalAveragePooling2D()(tensor_img)
        tensor_img = Dense(XXLG_CNT, activation='relu')(tensor_img)
        tensor_img = Dropout(DROPOUT_RATE)(tensor_img)

        return Dense(self.plant_species_cnt, activation='softmax')(tensor_img)

    def show_prediction_visualization(
            self,
            target_indices: Optional[List[int]] = None
    ) -> Tuple[int, int]:
        """
        Orchestrates high-performance batch predictions, isolates confidence metrics,
        and visually prints classification results with exact image bounds.
        """
        image_width, image_height, image_channels = self.image_params
        correct_cnt = 0
        image_cnt = len(self.x_test_norm)

        # Safe fallback bounds check for tracking parameters
        #batch_size = globals().get('TRAINED_BATCH_SIZE', 32)
        #rows_to_show = globals().get('IMAGE_ROWS', 4)

        print('\n* Generating Global Evaluation Slice Predictions... *')
        all_predicted_probs = self.model.predict(self.x_test_norm, batch_size=TRAINED_BATCH_SIZ, verbose=0)
        all_predicted_indices = np.argmax(all_predicted_probs, axis=1)
        all_true_indices = np.argmax(self.y_test_enc, axis=1)

        # Determine indices to visualize
        if target_indices is not None:
            indices_to_run = target_indices
            show_cnt = len(indices_to_run)
        elif self.all_pred:
            indices_to_run = list(range(image_cnt))
            show_cnt = image_cnt
        else:
            show_cnt = IMAGE_ROWS
            indices_to_run = random.sample(range(image_cnt), show_cnt)

        # Unified Loop Processing Step
        for i, index in enumerate(indices_to_run):
            print('# --- [START] --- #')
            print(f'--- [{i + 1} of {show_cnt}] ---')
            print(f'Dataset Random Index: {index}')

            predicted_class_index = all_predicted_indices[index]
            true_label_index = all_true_indices[index]

            print(f'predicted_class_index: {predicted_class_index}, true_label_index: {true_label_index}')

            # Map index values cleanly using global state bindings
            predicted_label = self._encoder.inverse_transform([predicted_class_index])[0]
            true_label = self._encoder.inverse_transform([true_label_index])[0]

            # ====================================================================
            # DIAGNOSTIC 1: TOP-3 CONFIDENCE EVALUATION BLOCK
            # ====================================================================
            predicted_probs_single = all_predicted_probs[index]
            top_3_indices = np.argsort(predicted_probs_single)[-3:][::-1]

            print(f"\n=== DEBUG: TOP 3 CONFIDENCE SCORES FOR INDEX {index} ===")
            for rank, idx in enumerate(top_3_indices):
                lbl = self._encoder.inverse_transform([idx])[0]
                conf = predicted_probs_single[idx] * 100
                marker = "⭐️ (Prediction)" if rank == 0 else ("🔹" if idx == true_label_index else "  ")
                print(f"  {marker} Rank {rank + 1}: Class {idx:2d} [{lbl:25s}] -> Confidence: {conf:.2f}%")
            print("========================================================\n")

            # ====================================================================
            # DIAGNOSTIC 2: IMAGE CHANNEL RGB vs BGR COLOR CHECK
            # ====================================================================
            sample_img = self.x_test_norm[index]
            print(f"=== DEBUG: IMAGE COLOR CHANNEL SIGNATURE (INDEX {index}) ===")
            print(f"  Raw Matrix Shape: {sample_img.shape} | Datatype: {sample_img.dtype}")
            print(f"  Channel 0 (R) Mean Intensity: {np.mean(sample_img[:,:,0]):.2f}")
            print(f"  Channel 1 (G) Mean Intensity: {np.mean(sample_img[:,:,1]):.2f}")
            print(f"  Channel 2 (B) Mean Intensity: {np.mean(sample_img[:,:,2]):.2f}")
            print("========================================================\n")

            plt.figure(num=f"Predictions (Index {index})", figsize=(4, 4))

            # Include explicit structural layout dimension tags
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

        return correct_cnt, show_cnt

    def show_results(self) -> None:
        """
        Orchestrates the visualization of model performance.
        """
        if self.eda:
            pass  # Future implementation block for show_plot_confusion_matrix

        predictions_correct, total = self.show_prediction_visualization()
        pct = (predictions_correct / total) * 100
        
        show_banner(f'{self.title} Results')
        print(f'{predictions_correct} out of {total} correct.\nAccuracy: {pct:.2f}%')

        if predictions_correct == total:
            print('\n⭐ PREDICTOR EARNED A PERFECT SCORE!  ⭐\n')