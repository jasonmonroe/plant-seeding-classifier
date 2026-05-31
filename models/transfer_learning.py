# models/transfer_learning.py

import random
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

from models.vgg import VggModel
from models.cnn_model import CnnModel
from src.config import (
    DROPOUT_RATE,
    IMAGE_PX_MAX,
    IMAGE_ROWS,
    FINE_TUNE_EPOCH_CNT,
    TL_LEARNING_RATE,
    TL_FINE_TUNE_LEARNING_RATE,
    TRAINED_BATCH_SIZE,
    XXLG_CNT,
    WARMUP_EPOCH_CNT
)

from src.eda import show_plot_confusion_matrix
from src.utils import show_banner, show_timer, start_timer

class TransferLayerModel(CnnModel):
    """
    Transfer Learning Model

    Leverages pre-trained ImageNet feature weights from a frozen VGG16 base
    to accurately classify target plant seedling images.
    """
    def __init__(self, vgg16_model: VggModel, dataset: dict, eda: bool, all_pred: bool):
        super().__init__(dataset=dataset)

        self.title = 'Transfer Learning Model'
        self.optimizer = Adam(learning_rate=TL_LEARNING_RATE)
        self.image_params = vgg16_model.image_params
        self.eda = eda
        self.all_pred = all_pred

        # Keep a tracking reference to the base model block
        self.vgg16_base_model = vgg16_model.get_base()
        self._create()

    def _create(self) -> None:
        print(f'\n--- 🪄 Creating {self.title} ---')

        inputs = Input(shape=self.image_params)
        outputs = self.__build_outputs(inputs)
        self.model = Model(inputs, outputs, name='vgg16_refined')

    def __build_outputs(self, inputs: tf.Tensor) -> Dense:
        # Logic: VGG16 preprocess_input expects 0-255 range BGR data.
        # We check the max of the input tensor; if <= 1.0, we assume it's normalized 
        # and scale it up. If > 1.0, we treat it as raw pixels.
        tensor_img = layers.Lambda(
            lambda img: preprocess_input(
                tf.where(tf.reduce_max(img) <= 1.0, img * IMAGE_PX_MAX, img)
            ),
            name='vgg16_preprocess_robust'
        )(inputs)

        # 2. Forward pass through VGG base (Keep training=False to protect batch-norm layers)
        tensor_img = self.vgg16_base_model(tensor_img, training=False)

        # Use both Average and Max pooling to catch textures AND edges
        avg_pool = GlobalAveragePooling2D()(tensor_img)
        max_pool = GlobalMaxPooling2D()(tensor_img)
        tensor_img = Concatenate()([avg_pool, max_pool])
        
        # Added BatchNormalization to stabilize the head and prevent loss explosion
        tensor_img = BatchNormalization()(tensor_img)
        
        # Double the density for the merged pooling layers
        tensor_img = Dense(XXLG_CNT, activation='relu')(tensor_img)
        tensor_img = BatchNormalization()(tensor_img)
        tensor_img = Dropout(DROPOUT_RATE)(tensor_img)

        return Dense(self.plant_species_cnt, activation='softmax', name='prediction_layer')(tensor_img)

    def fine_tune(self, unfreeze_block: str, fine_tune_lr: float) -> None:
        """
        Surgically unfreezes specific convolutional blocks of VGG16
        and re-compiles the system graph cleanly for target classification.
        """
        print(f"\n--- 🔥Initiating Fine-Tuning: Unfreezing {unfreeze_block} 🔥 ---")

        # Unfreeze the base convolutional block layers
        self.vgg16_base_model.trainable = True
        for layer in self.vgg16_base_model.layers:
            if unfreeze_block in layer.name:
                layer.trainable = True
                print(f" -> 🔥 Unfrozen: {layer.name}")
            else:
                layer.trainable = False

        # Recompile using a specialized low learning rate to protect pre-trained features
        self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr), 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit_model(self, generator: NumpyArrayIterator, class_weights: dict, epoch_cnt: int) -> History:
        # Overrides cnn_model.fit_model()
        return self.model.fit(
            batch_size=TRAINED_BATCH_SIZE,
            callbacks=[self._reduce_lr, self._early_stopping],
            class_weight=class_weights,
            epochs=epoch_cnt,
            validation_data=(self.x_val_norm, self.y_val_enc),
            verbose=1,
            x=generator
        )

    def show_prediction_visualization(self, target_indices: Optional[List[int]] = None) -> Tuple[int, int]:
        correct_cnt = 0
        image_cnt = len(self.x_test_norm)

        all_predicted_probs = self.model.predict(self.x_test_norm, batch_size=TRAINED_BATCH_SIZE, verbose=0)
        all_predicted_indices = np.argmax(all_predicted_probs, axis=1)
        all_true_indices = np.argmax(self.y_test_enc, axis=1)

        if target_indices is not None:
            indices_to_run = target_indices
        elif self.all_pred:
            indices_to_run = list(range(image_cnt))
        else:
            indices_to_run = random.sample(range(image_cnt), IMAGE_ROWS)

        show_cnt = len(indices_to_run)
        print(f'\n* Generating {show_cnt} Sample Visualizations *')

        for i, index in enumerate(indices_to_run):
            pred_idx = all_predicted_indices[index]
            true_idx = all_true_indices[index]

            predicted_label = self._encoder.inverse_transform([pred_idx])[0]
            true_label = self._encoder.inverse_transform([true_idx])[0]

            print(f'\n--- [{i + 1} of {show_cnt}] (Index: {index}) ---')

            if pred_idx != true_idx:
                probs = all_predicted_probs[index]
                top_3 = np.argsort(probs)[-3:][::-1]
                print(f'⚠️ Top 3 Confidence Scores:')
                for rank, idx in enumerate(top_3):
                    lbl = self._encoder.inverse_transform([idx])[0]
                    marker = "⭐️" if rank == 0 else ("🔹" if idx == true_idx else "  ")
                    print(f"\t{marker} Rank {rank + 1}: {lbl:22s} -> {probs[idx]*100:.2f}%")

            plt.figure(num=f"Predictions (Index {index})", figsize=(4, 4))
            plt.xlabel(f"Width: {self.image_params[0]}px")
            plt.ylabel(f"Height: {self.image_params[1]}px")

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

            if pred_idx == true_idx:
                print('✅ Correct Prediction!')
                correct_cnt += 1
            else:
                print('❌ Incorrect Prediction')

        return correct_cnt, show_cnt

    def show_results(self) -> None:

        if self.eda:
            show_plot_confusion_matrix()

        print(f'\n--- {self.title} Classification Report ---')
        self.print_classification_report(self.model, self.x_test_norm, self.y_test_enc)

        predictions_correct, total = self.show_prediction_visualization()
        pct = (predictions_correct / total) * 100

        print(f'\n--- {self.title} Results ---')
        print(f'{predictions_correct} out of {total} correct.\nAccuracy: {pct:.2f}%')

        if predictions_correct == total:
            print('\n⭐ PREDICTOR EARNED A PERFECT SCORE! ⭐\n')

    def run(self, train_generator: NumpyArrayIterator):
        self.compile()
        self.show_summary()
        show_banner(self.title, 'Fitting Training Model...')

        # Fetch class weights from parent logic to handle dataset imbalance
        class_weights = self._get_class_weights()

        print('\n--- 📢 Phase #1: Training Classification Head (Base Locked) ---\n')

        # Reduced warmup to prevent head-overfitting
        start_time = start_timer()
        history_warmup = self.fit_model(
            generator=train_generator,
            class_weights=class_weights,
            epoch_cnt=WARMUP_EPOCH_CNT
        )
        show_timer(start_time)

        self.fine_tune(unfreeze_block='block5', fine_tune_lr=TL_FINE_TUNE_LEARNING_RATE)

        print('\n--- 📣 Phase #2: Fine-Tuning Deep Convolutional Weights ---\n')
        
        start_time = start_timer()
        history_finetune = self.fit_model(
            generator=train_generator,
            class_weights=class_weights,
            epoch_cnt=FINE_TUNE_EPOCH_CNT
        )
        show_timer(start_time)

        # Capture the final state for the FinalReport
        self.history = history_finetune

        if self.eda:
            self.show_history()
        
        self.evaluate(verbose=0) # Update self.loss and self.accuracy
        self.calc_performance()
        self.get_predictions()
        self.show_results()
