# modeling.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


 

# Note: Assume label_encoder() is instantiated
def encode_label(data: pd.DataFrame, label_encoder: LabelEncoder, plant_species_cnt: int) -> np.ndarray:
    # Encode categorical features and scale the pixel values
    # Creating one-hot encoded representation of target labels

    return tf.keras.utils.to_categorical(
        label_encoder.fit_transform(data),
        num_classes=plant_species_cnt
    )

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# Used to one-hot encode integer labels into a binary matrix representation that's
# suitable for classification tasks in deep learing.
def encode_data(y_training_data: pd.DataFrame, y_testing_data: pd.DataFrame, y_validation_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    y_training_encoded = encode_label(y_training_data)
    y_testing_encoded = encode_label(y_testing_data)
    y_validation_encoded = encode_label(y_validation_data)

    return y_training_encoded, y_testing_encoded, y_validation_encoded

def get_model_predictions(mod: Sequential, x_training_normalized: np.ndarray, x_testing_normalized: np.ndarray, y_testing_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_predictor_training = mod.predict(x_training_normalized)
    y_predictor_testing = mod.predict(x_testing_normalized)
    y_testing_normalized = np.argmax(y_testing_encoded, axis=1)

    return y_predictor_training, y_predictor_testing, y_testing_normalized

# Evaluate Model and return the accuracy and loss analysis
def evalute_model(mod: Sequential, x_testing_normalized: np.ndarray, y_testing_encoded: np.ndarray) -> Tuple[float, float]:
    loss, accuracy = mod.evaluate(
        x_testing_normalized,
        y_testing_encoded,
        verbose=2
    )

    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    return loss, accuracy

# Fit Model and return history.  Also time the optimization.
def fit_model(mod: Sequential, x_training_normalized: np.ndarray, y_training_encoded: np.ndarray, x_validation_normalized: np.ndarray, y_validation_encoded: np.ndarray) -> History:

    y_training_classes = np.argmax(y_training_encoded, axis=1)
    raw_class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_training_classes),
        y=y_training_classes
    )
    # Convert the array of weights to a dictionary mapping class indices to weights
    class_weights_dict = dict(enumerate(raw_class_weights))

    return mod.fit(
        x_training_normalized,
        y_training_encoded,
        epochs=BASE_EPOCH_CNT,
        validation_data=(
            x_validation_normalized,
            y_validation_encoded
        ),
        batch_size=BASE_BATCH_SIZE,
        verbose=2,
        class_weight=class_weights_dict,
        #callbacks=[reduce_lr, early_stopping]
    )

# Fit Model with trained generated data
def fit_trained_model(
        mod: Sequential,
        x_training_normalized: np.ndarray,
        y_training_encoded: np.ndarray,
        x_validation_normalized: np.ndarray,
        y_validation_encoded: np.ndarray,
        datagen: ImageDataGenerator,
        validation_generator: NumpyArrayIterator,
        reduce_lr: ReduceLROnPlateau,
        early_stopping: EarlyStopping
) -> History:

    return mod.fit(
        datagen.flow(
            x_training_normalized,
            y_training_encoded,
            batch_size=TRAINED_BATCH_SIZE,
            seed=SEED,
            shuffle=True,
        ),
        epochs=TRAINED_EPOCH_CNT,
        steps_per_epoch=(x_training_normalized.shape[0] // TRAINED_BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=(x_validation_data.shape[0] // TRAINED_BATCH_SIZE),
        verbose=1,
        callbacks=[reduce_lr, early_stopping]
    )


# ==================================
#  MODEL PERFORMANCE CLASSIFICATION
# ==================================
#
# Metric, Working (Minimum Signal), Good (Expected Target), Excellent (Production Goal)
# Test Accuracy, 60%−75%, 75%−85%, >85%
# Test F1-Score, 0.55−0.70, 0.75−0.85, >0.85
# Test Loss, <1.5, <0.7, <0.4
#
# Define a function to compute different metrics to check performance of a classification model built using stats models

def model_performance_classification(mod: Sequential, predictors: np.ndarray, target: str, threshold: float=0.5) -> pd.DataFrame:

    """
    Function to compute different metrics to check classification model performance
    model: classifer
    predictors: independent variables
    target: target variable
    threshold: threshold for classification
    """

    # Use np.argmax to get the predicted class index
    # (i.e: predictors ~ x_training_normalized)
    predicted_labels = np.argmax(mod.predict(predictors), axis=1)

    # Convert TRUE one-hot encoded target labels to integer class indices
    # (i.e: target ~ y_training_encoded)
    target_classes = np.argmax(target, axis=1)

    acc = accuracy_score(target_classes, predicted_labels)
    prec = precision_score(target_classes, predicted_labels, average='weighted')
    rec = recall_score(target_classes, predicted_labels, average='weighted')
    f1 = f1_score(target_classes, predicted_labels, average='weighted')

    perform_df = pd.DataFrame({'Accuracy': [acc], 'Precision': [prec], 'Recall': [rec], 'F1': [f1]})

    return perform_df


# Visualizing the predicted and correct label of images from test data, add `pixel height`, `pixel width` labels
def show_visualize_prediction(
        mod: Sequential,
        enc: LabelEncoder,
        x_test: np.ndarray,
        x_testing_norm: np.ndarray,
        y_testing_enc: np.ndarray,
        show_all: bool=False
) -> Tuple[int, int]:

    correct_cnt = 0
    image_cnt = len(x_testing_norm)
    show_cnt = IMAGE_ROWS

    if show_all:
        show_cnt = image_cnt

    for i in range(0, show_cnt):
        if show_all:
            index = i
        else:
            index = random.randint(0, image_cnt)

        print(f'# ----- [{i} of {show_cnt}] ----- #')
        print(f'Index:{index}\n')

        # Get the predicted probabilities
        predicted_probs = mod.predict((x_testing_norm[index].reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predicted_probs, axis=1)

        # Use the index to get the predicted label
        predicted_label = enc.inverse_transform(predicted_class_index)
        predicted_label = predicted_label[0]

        # Using inverse_transform() to get the output label from the output vector.
        true_label_index = np.argmax(y_testing_enc[index])  # Get index of true label
        true_label = enc.classes_[true_label_index]         # Get true label from index

        plt.figure(figsize=(2, 2))
        plt.imshow(x_test[index])
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

def print_classification_report(mod, x_data: np.ndarray, y_true_encoded: np.ndarray):
    y_true_labels = np.argmax(y_true_encoded, axis=1)
    y_pred_probs = mod.predict(x_data)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print(classification_report(
        y_true_labels,
        y_pred_classes,
        target_names=plant_species,
        digits=4)
    )