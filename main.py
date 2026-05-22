# main.py
import sys
import warnings
import random


import numpy as np
import pandas as pd

import cv2 # OpenCV for image processing

from sklearn.preprocessing import LabelEncoder

# TensorFlow and Keras libraries
import tensorflow as tf



from tensorflow.keras import backend
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D, # Import the GlobalAveragePooling2D layer
    MaxPooling2D,
)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    NumpyArrayIterator # The iterator is generally created by ImageDataGenerator
)

from models import base
from models.base import BaseModel
from models.data_augm import DataAugmentedModel
from models.vgg import Vgg
from notebooks.plant_seed_classification import training_generator
from src import image_handler
#from src import image_handler, data_handler

# Local Source Files
from src.config import DA_LEARNING_RATE, L2_LEARNING_RATE, LG_CNT, TL_LEARNING_RATE
from src.data_handler import DataHandler
from src.image_handler import ImageHandler
from models.modeler import Modeler
from models.cnn_model import CnnModel
from src.modeling import print_classification_report
from src.preprocess import load_data, load_images, describe_data, describe_images, describe_labels, split_data
#from src.modeling import create_base_model, create_data_augmented_model, create_transfer_learning_model, create_vgg_model, fit_model, evalute_model, fit_trained_model, model_performance_classification, get_model_predictions, encode_data, encode_label, print_classification_report, show_visualize_prediction
#from src.imaging import build_generator, generate_training_image_batch, show_augmented_image_batch, show_random_image, create_bgr_images, convert_to_rgb, show_random_cv2_image, show_raw_images, show_augmented_image_batch, get_resized_images, normalize
from src.utils import init_cnn_session, get_plant_species, show_banner, start_timer, show_timer
from src.eda import show_plot_confusion_matrix, show_plot_history, show_plot_histogram, show_plot_confusion_matrix, show_plant_species_dist, show_labeled_barplot

def run_main_pipeline():

    # Fix warnings
    warnings.filterwarnings('ignore')

    #CUDA_LAUNCH_BLOCKING = 1
    print("Number of GPU's Available: ", len(tf.config.list_physical_devices('GPU')))

    # Initialize Data Handler for Plant Seedlings
    plant = DataHandler()

    plant.describe_data()
    plant.describe_label()
    df_labels = plant.get_labels()

    # Get Plant Species
    plant_species = get_plant_species(df_labels)
    plant_species_cnt = len(plant_species)
    print(f'\nPlant Species Count: {plant_species_cnt}')

    # ==================================
    #  IMAGE INFORMATION
    # ==================================
    # mean, median, std dev, min, max

    print('Image Information')
    image_width, image_height, image_channels = plant.describe_images()
    images = plant.get_images()

    describe_images(images)

    # Randomly sample pixel data
    image_handle = ImageHandler(images)
    random_img = image_handle.show_random_image(images)

    # Plot histogram to check distribution
    show_plot_histogram(images)

    """
    Most of the pixels RGB values are between 50-100 with the peak around 75.  This means most of the image data and 
    value is in the middle.
    """

    # Check for NaN values
    print(f'There are {np.isnan(images).sum()} NaN values in the dataset.')

    """
    Exploratory Data Analysis
    
    - EDA is an important part of any project involving data.
    - It is important to investigate and understand the data better before building a model with it.
    - A few questions have been mentioned below which will help you understand the data better.
    - A thorough analysis of the data, in addition to the questions mentioned below, should be done.
    
    1. How are these different category plant images different from each other?
    2. Is the dataset provided an imbalance? (Check with using bar plots)
    """

    # Plot images like a grid.
    labels = plant.get_labels(True)
    image_handle.show_raw_images(labels)
    #show_raw_images(images, labels)

    # Label barplot with label
    show_labeled_barplot(df_labels, 'Label', perc=True)

    """
    Observations:
    These are the 12 labels of images.  
    
    1. Data is not even.  Some labels are less than others meaning there are fewer types of plants in the sample set.
    2. If all labels were even the average would be 8.3% a piece.  Anything over is over represented.  Anything under 
    is under represented.
    3. Charlock is the closest label to the average.
    4. Loose Silky-Ben has the most labels with 13.8%.
    5. Maize and Common wheat are tied for last at 4.7%.
    """
    
    # eda.py 
    show_plant_species_dist(df_labels)

    # Data Pre-Processing - Convert the BGR images to RGB images.

    # Convert the BGR images to RGB images.
    # First, we will display the image as it is imported which means in BGR format.

    #show_random_cv2_image(bgr_images)
    image_handle.show_random_cv2_image()
    bgr_images = image_handle.create_bgr_images()

    # Now to convert BGR to RGB
    #rgb_images = convert_to_rgb(bgr_images)
    #show_random_cv2_image(rgb_images)
    rgb_images = image_handle.convert_to_rgb(bgr_images)

    """
    RGB Image (that has been converted).
    Resize the images
    As the size of the images is large, it may be computationally expensive to train on these larger images; therefore, 
    it is preferable to reduce the image size from 128 to 64.
    
    Note: Your scores will reduce if you lower image size. So it's your choice.
    """

    # Resize RGB images
    # Note: This will be used as `independent variables`.

     # You can reduce the image in half but for now we will keep same size due to original images having a small filesize.
    reduce_by = 1 # Note: was 2
    resized_img_dims = (image_height // reduce_by, image_width // reduce_by)
    resized_images = image_handle.get_resized_images(rgb_images, resized_img_dims)

    # Resized Image Below
    random_img = image_handle.show_random_image(resized_images)

    # Check if that image was resized
    image_params = (image_height, image_width, image_channels)
    if image_handle.is_resized(random_img, resized_img_dims):
        image_dims = resized_img_dims
        image_params = image_dims + (image_channels,)


    # Define reduce liner regression and early stopping objects
    # Define ReduceLRonPlateau()
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=L2_LEARNING_RATE,
        verbose=1
    )

    """
    Early Stopping
    
    Monitor validation loss and stop training automatically when the loss starts consistently increasing as a sign of 
    overfitting.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    """
    Data Preparation for Modeling
    
    - Before you proceed to build a model, you need to split the data into train, test, and validation to be able to evaluate the model that you build on the train data
    - You'll have to encode categorical features and scale the pixel values.
    - You will build a model using the train data and then check its performance
    
    Split the dataset
    """

    #x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data = split_data(resized_images, df_labels['Label'])

    (x_training_data,
     y_training_data,
     x_validation_data,
     y_validation_data,
     x_testing_data,
     y_testing_data) = plant.split_data(resized_images)

    # Load parent model class for data preparation

    #modeler = Modeler(plant_species_cnt, x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data)


    # Build Cnn Model
    cnn_model = CnnModel(
        plant_species,
        x_training_data,
        y_training_data,
        x_validation_data,
        y_validation_data,
        x_testing_data,
        y_testing_data,
    )

    cnn_model.encode_data()
    cnn_model.normalize()

    # --- Build Models --- #
    """
    🧠 What is a CNN Model?
    A Convolutional Neural Network (CNN), or ConvNet, is a specialized type of deep learning model designed primarily 
    to process data that has a known grid-like topology, such as images (2D grid of pixels) or time-series data 
    (1D grid).
    """
    base_model = BaseModel(image_params)

    # Note: base model created...
    base_model.compile()
    base_model.show_summary()

    # Calculate the number of parameters.
    param_cnt = base_model.model.count_params()
    print(f'Number of parameters: {param_cnt}')

    start_time = start_timer()
    base_model.fit_model()

    show_timer(start_time)
    show_plot_history(base_model.history, base_model.title, 'accuracy')
    show_plot_history(base_model.history, base_model.title, 'loss')

    """
    Observations:
    
    By comparing the final Training Accuracy ($90.53%) and Validation Accuracy ($80.63%), we 
    can draw two conclusions:
    Final Accuracy Gap: There is an $9.90% gap between the training and validation accuracy 
    ($90.53%} - 80.63%).This confirms that the model is overfitting, meaning it learned the 
    training data much better than the validation data. However, a $10%$ gap is manageable and a strong improvement 
    from earlier runs.
    
    Best Epoch: The peak validation accuracy of $82.32% was hit at Epoch 24 before dropping slightly at 
    the end. The validation loss ($0.7476) was the lowest at Epoch 30, indicating a stable final state.
    This training history is successful because the Base CNN Model achieved an excellent generalization accuracy of 
    over 80%, which is why it performed so well on the final testing set.
    
    The Results are Excellent (for this model):
    *   Accuracy is higher for training then validation.
    *   Closest the data comes together is after the 8th epoch.
    *   Loss data decreases linearly.
    *   Training loss diminishes more extremely than validation losses as epochs increase.  In other words it more 
        volatile dealing with the training loss.
    """

    # Evaluate CNN Model
    start_time = start_timer()
    base_model.evaluate()
    show_timer(start_time)

    """
    🎯 Interpretation
    
    High Generalization: This 84.00 Test Accuracy is the most important number. When compared to the 
    Training Accuracy of 90.53 (from the training log), the model maintained strong performance on new 
    data. This shows the model successfully generalized and did not severely overfit.
    
    * Model Stability: The Test Loss of $0.7909 is relatively low, confirming that the model's final weights are stable 
    and effective at making accurate probability assignments.
    * Efficiency: The model is highly efficient, evaluating the entire test set in less than half a second.
    
    Conclusion:
    
    This evaluation confirms that the Base CNN Model is a successful and robust solution for the Plant Seed 
    Classification task, meeting your performance goals.
    """

    # Model Performance Classification
    base_model.calc_performance()
    _, y_test_pred, _ = base_model.get_predictions()

    # Plotting Confusion Matrix
    show_plot_confusion_matrix(base_model.y_test_enc, y_test_pred)
    show_banner(base_model.title, 'Classification Report')
    print_classification_report(base_model.model, base_model.x_test_norm, base_model.y_test_enc, plant_species)

    """
    Data Augmentation
    
    Note: Data augmentation should not be used in the validation/test data set.
    
    CNN Model with Data Augmentation
    
    The purpose of using a Data Augmented Model (DAM)—or more accurately, applying data augmentation during training—is 
    to artificially increase the size and diversity of your training data without collecting new physical images.
    """

    # Data Augmented CNN Model

    data_augm_model = DataAugmentedModel('Data Augmented CNN Model', image_params)
    data_augm_model.compile()
    data_augm_model.show_summary()

    start_time = start_timer()
    show_banner(data_augm_model.title, 'Fitting Training Model')

    # Augment the data without using validation or test data.  Only training data.
    # The rescale=1./IMAGE_PX_MAX is removed as data is already normalized.
    train_datagen = image_handle.generate_training_image_batch()
    train_generator = image_handle.build_generator(train_datagen, base_model.x_train_norm, base_model.y_train_enc)

    # The rescale=1./IMAGE_PX_MAX is removed as data is already normalized.
    val_datagen = ImageDataGenerator()
    val_generator = image_handle.build_generator(
        val_datagen,
        base_model.x_val_norm,
        base_model.y_val_enc,
        shuffle_flag=False
    )

    data_augm_model.fit_trained_model(train_datagen, val_generator)
    show_timer(start_time)

    """
    "The training accuracy starts low and increases, but the final score is much lower than the $90% seen in the Base 
    CNN.
    
    The validation accuracy reached a peak of $70.95% at Epoch 27, which aligns with the final 
    $71.58% testing accuracy you reported for this model."

    Key Observations and Callback Activity
    
    Training vs. Validation: 
    Throughout the run, the Validation Accuracy is often higher than the Training Accuracy
     (e.g., Epoch 27: Train approx 66%, Val $70.95%). This is the signature of strong regularization (via data 
     augmentation), which prevented the model from overfitting.
    
    Learning Rate Reduction: 
    The ReduceLROnPlateau callback triggered twice:
    
    Epoch 6: Learning Rate reduced from $1 \\times 10^{-4 to $5 \\times 10^{-5 because the validation metrics 
    plateaued or failed to improve.
    
    Epoch 36: Learning Rate reduced again from $5 \\times 10^{-5 to $2.5 \\times 10^{-5. This is a common strategy 
    to help the model escape local minima and continue learning, even if very slowly.
    
    
    Conclusion: This training log confirms the model was stable and effectively prevented overfitting, but the 
    difficulty in learning the highly augmented training data resulted in a lower final performance (the 
    71.58% test accuracy).
    """

    # Look at the images after data has been augmented
    show_plot_history(data_augm_model.history, data_augm_model.title, 'accuracy')
    show_plot_history(data_augm_model.history, data_augm_model.title, 'loss')

    """
    Observations:
    
    * The results are unexpected.  The validation accuracy is static until the 5th epochs and then climbs .10% by the 
    8th epochs and then tepid's off.
    * Train accuracy bounces up and down before the accuracy increases while validation accuracy increases.
    * Validation accuracy increases as the epochs increase which means it's learning more per epoch. (This is what we 
    want).
    * Training accuracy is a hit or miss but eventually goes up after the 25th epoch.  We need better results.
    * The losses are closely aligned by each epoch.  The training losses still bounce up and down but overall decreases 
    each epoch.
    """

    # Evaluate the CNN Model w/ Data Augmentation
    start_time = start_timer()
    show_banner(data_augm_model.title, 'Evaluation')
    data_augm_model.evaluate()
    show_timer(start_time)

    # Get CNN Model w/ Data Augmentation training performance
    data_augm_model.calc_performance()
    _, y_test_pred, _ = data_augm_model.get_predictions()
    show_plot_confusion_matrix(data_augm_model.y_test_enc, y_test_pred)
    show_banner(data_augm_model.title, 'Classification Report')
    print_classification_report(data_augm_model.model, data_augm_model.x_test_norm, data_augm_model.y_test_enc, plant_species)

    # VGG16 Model
    image_params = (LG_CNT, LG_CNT, 3)
    vgg_model = Vgg(image_params)
    vgg_model.show_summary()


    # Transfer Learning Model



    # ---> @todo First Draft Below <--- #

    """
    Encode the target labels
    """

    # Encode categorical features and scale the pixel values
    # Creating one-hot encoded representation of target labels
    label_encoder = LabelEncoder()

    y_training_encoded, y_testing_encoded, y_validation_encoded = encode_data(
        label_encoder,
        y_training_data,
        y_testing_data,
        y_validation_data,
        plant_species_cnt
    )

    """#Data Normalization"""

    # Normalizing the image data
    x_training_normalized = normalize(x_training_data)
    x_testing_normalized = normalize(x_testing_data)
    x_validation_normalized = normalize(x_validation_data)


    # --- Model Building --- #

    """Model Building
    
    ###CNN Model
    
    🧠 What is a CNN Model?
    A Convolutional Neural Network (CNN), or ConvNet, is a specialized type of deep learning model designed primarily to process data that has a known grid-like topology, such as images (2D grid of pixels) or time-series data (1D grid).
    """

    init_cnn_session()

    # ==================================
    #  CNN Model: Base Model
    # ==================================
    # Intializing a sequential CNN model

    base_model_title = 'Base CNN Model'
    base_model = create_base_model(plant_species_cnt, image_params)

    # Compile Model
    base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Get Model Summary
    base_model.summary()

    # Calculate the number of parameters.
    param_cnt = base_model.count_params()
    print(f'Number of parameters: {param_cnt}')

    # Fit CNN Model and return history.  Also time the optimization.
    start_time = start_timer()
    show_banner(base_model_title, 'Fitting Base Model...')

    base_model_history = fit_model(
        base_model,
        x_training_normalized,
        y_training_encoded,
        x_validation_normalized,
        y_validation_encoded
    )

    show_timer(start_time)

    # Plot History
    show_plot_history(base_model_history, base_model_title, 'accuracy')
    show_plot_history(base_model_history, base_model_title, 'loss')

    """
    Observations:
    
    By comparing the final Training Accuracy ($90.53%) and Validation Accuracy ($80.63%), we 
    can draw two conclusions:
    Final Accuracy Gap: There is an $9.90% gap between the training and validation accuracy 
    ($90.53%} - 80.63%).This confirms that the model is overfitting, meaning it learned the 
    training data much better than the validation data. However, a $10%$ gap is manageable and a strong improvement 
    from earlier runs.
    
    r"Best Epoch: The peak validation accuracy of $82.32% was hit at Epoch 24 before dropping slightly at 
    the end. The validation loss ($0.7476) was the lowest at Epoch 30, indicating a stable final state.
    This training history is successful because the Base CNN Model achieved an excellent generalization accuracy of 
    over 80%, which is why it performed so well on the final testing set.
    
    The Results are Excellent (for this model):
    *   Accuracy is higher for training then validation.
    *   Closest the data comes together is after the 8th epoch.
    *   Loss data decreases linearly.
    *   Training loss diminishes more extremely than validation losses as epochs increase.  In other words it more 
        volatile dealing with the training loss.
    """

    # Evaluate CNN Model
    start_time = start_timer()
    show_banner(base_model_title, 'Evaluation')

    base_model_loss, base_model_accuracy = evalute_model(
        base_model,
        x_testing_normalized,
        y_testing_encoded
    )

    show_timer(start_time)

    """🎯 
    Interpretation
    
    r"High Generalization: This 84.00 Test Accuracy is the most important number. When compared to the 
    Training Accuracy of 90.53 (from the training log), the model maintained strong performance on new 
    data. This shows the model successfully generalized and did not severely overfit.
    
    * Model Stability: The Test Loss of $0.7909 is relatively low, confirming that the model's final weights are stable 
    and effective at making accurate probability assignments.
    * Efficiency: The model is highly efficient, evaluating the entire test set in less than half a second.
    
    Conclusion:
    
    This evaluation confirms that the Base CNN Model is a successful and robust solution for the Plant Seed 
    Classification task, meeting your performance goals.
    """

    # --- Model Performance ---

    # Get model training performance
    base_training_perf = model_performance_classification(base_model, x_training_normalized, y_training_encoded)
    print(base_training_perf)

    y_predictor_training, y_predictor_testing, y_testing_normalized = get_model_predictions(
        base_model,
        x_training_normalized,
        x_testing_normalized,
        y_testing_encoded
    )

    # Plotting confusion matrix
    show_plot_confusion_matrix(y_testing_encoded, y_predictor_testing)

    show_banner(base_model_title, 'Classification Report')
    print_classification_report(base_model, x_testing_normalized, y_testing_encoded, plant_species)


    """
    # Define ReduceLRonPlateau()
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=L2_LEARNING_RATE,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    """

    # Augment the data without using validation or test data.  Only training data.
    # The rescale=1./IMAGE_PX_MAX is removed as data is already normalized.
    """
    training_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    """

    training_datagen = image_handle.generate_training_image_batch()

    # Flowing training images in batches of 48 using training_datagen generator.
    training_generator = build_generator(training_datagen, x_training_normalized, y_training_encoded)
    
    """
    training_generator = training_datagen.flow(
        x_training_normalized,
        y_training_encoded,
        batch_size=GENERATOR_BATCH_SIZE,
        seed=SEED,
        shuffle=True
    )
    """

    # The rescale=1./IMAGE_PX_MAX is removed as data is already normalized.
    validation_datagen = ImageDataGenerator() # empty params
    validation_generator = build_generator(validation_datagen, x_validation_normalized, y_validation_encoded, shuffle_flag=False)


    """
    validation_generator = validation_datagen.flow(
        x_validation_normalized,
        y_validation_encoded,
        batch_size=GENERATOR_BATCH_SIZE,
        seed=SEED,
        shuffle=False
    )
    """

    """
    Data Augmentation
    
    Remember, data augmentation should not be used in the validation/test data set.
    
    CNN Model with Data Augmentation
    
    The purpose of using a Data Augmented Model (DAM)—or more accurately, applying data augmentation during training—is 
    to artificially increase the size and diversity of your training data without collecting new physical images.
    """

    init_cnn_session()

    da_model_title = 'Data Augmented CNN Model'

    """
    da_model = Sequential()

    # --- Block 1 ---
    da_model.add(Conv2D(SM_CNT, KERNEL_SIZE_MED, padding='same', input_shape=IMAGE_PARAMS))
    da_model.add(BatchNormalization())
    da_model.add(Activation('relu'))
    da_model.add(MaxPooling2D(pool_size=KERNEL_SIZE_SM))

    # --- Block 2 ---
    da_model.add(Conv2D(MED_CNT, KERNEL_SIZE_MED, padding='same'))
    da_model.add(BatchNormalization())
    da_model.add(Activation('relu'))
    da_model.add(MaxPooling2D(pool_size=KERNEL_SIZE_SM))

    # --- Block 3 ---
    da_model.add(Conv2D(LG_CNT, KERNEL_SIZE_MED, padding='same'))
    da_model.add(BatchNormalization())
    da_model.add(Activation('relu'))
    da_model.add(MaxPooling2D(pool_size=KERNEL_SIZE_SM))

    # --- Global Pooling and Dense Layers ---
    da_model.add(GlobalAveragePooling2D())
    da_model.add(Dense(XLG_CNT))
    da_model.add(BatchNormalization())
    da_model.add(Activation('relu'))
    da_model.add(Dropout(DROPOUT_RATE))
    da_model.add(Dense(plant_species_cnt, activation='softmax'))
    """

    da_model = create_data_augmented_model(plant_species_cnt, image_params)

    # Compile Data Augmented CNN Model
    da_model.compile(
        optimizer=Adam(learning_rate=DA_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Generate summary
    da_model.summary()

    start_time = start_timer()

    show_banner(da_model_title, 'Fitting Training Model')

    # Fit model with augmented data
    da_model_history = fit_trained_model(
        da_model,
        x_training_normalized,
        y_training_encoded,
        x_validation_normalized,
        y_validation_encoded,
        x_validation_data,
        training_datagen,
        validation_generator,
        reduce_lr,
        early_stopping
    )

    show_timer(start_time)

    """
    "The training accuracy starts low and increases, but the final score is much lower than the $90% seen in the Base 
    CNN.
    
    The validation accuracy reached a peak of $70.95% at Epoch 27, which aligns with the final 
    $71.58% testing accuracy you reported for this model."

    Key Observations and Callback Activity
    
    Training vs. Validation: 
    Throughout the run, the Validation Accuracy is often higher than the Training Accuracy
     (e.g., Epoch 27: Train approx 66%, Val $70.95%). This is the signature of strong regularization (via data 
     augmentation), which prevented the model from overfitting.
    
    Learning Rate Reduction: 
    The ReduceLROnPlateau callback triggered twice:
    
    Epoch 6: Learning Rate reduced from $1 \\times 10^{-4 to $5 \\times 10^{-5 because the validation metrics 
    plateaued or failed to improve.
    
    Epoch 36: Learning Rate reduced again from $5 \\times 10^{-5 to $2.5 \\times 10^{-5. This is a common strategy 
    to help the model escape local minima and continue learning, even if very slowly.
    
    
    Conclusion: This training log confirms the model was stable and effectively prevented overfitting, but the 
    difficulty in learning the highly augmented training data resulted in a lower final performance (the 
    71.58% test accuracy).
    """

    # Look at the images after data has been augmented
    show_plot_history(da_model_history, da_model_title, 'accuracy')
    show_plot_history(da_model_history, da_model_title, 'loss')

    """
    Observations:
    
    * The results are unexpected.  The validation accuracy is static until the 5th epochs and then climbs .10% by the 
    8th epochs and then tepid's off.
    * Train accuracy bounces up and down before the accuracy increases while validation accuracy increases.
    * Validation accuracy increases as the epochs increase which means it's learning more per epoch. (This is what we 
    want).
    * Training accuracy is a hit or miss but eventually goes up after the 25th epoch.  We need better results.
    * The losses are closely aligned by each epoch.  The training losses still bounce up and down but overall decreases 
    each epoch.
    """

    # Evaluate the CNN Model w/ Data Augmentation
    start_time = start_timer()
    show_banner(da_model_title, 'Evaluation')

    da_model_loss, da_model_accuracy = evalute_model(
        da_model,
        x_testing_normalized,
        y_testing_encoded
    )

    show_timer(start_time)

    # Get CNN Model w/ Data Augmentation training performance
    da_model_training_perf = model_performance_classification(
        da_model,
        x_training_normalized,
        y_training_encoded
    )

    print(da_model_training_perf)

    # 0	0.723158	0.734437	0.723158	0.709289

    y_predictor_training, y_predictor_testing, y_testing_normalized = get_model_predictions(
        da_model,
        x_training_normalized,
        x_testing_normalized,
        y_testing_encoded
    )

    # Show confusion matrix for augmented data
    show_plot_confusion_matrix(y_testing_encoded, y_predictor_testing)

    show_banner(da_model_title, 'Classification Report')
    print_classification_report(da_model, x_testing_normalized, y_testing_encoded, plant_species)

    """
    Final Model
     
    Comment on the final model you have selected and use the same in the below code to visualize the image.
    """

    init_cnn_session()

    # VGG16 Model

    image_params = (LG_CNT, LG_CNT, 3)

    # ==================================
    #  CNN Model: VGG16
    # ==================================
    vgg_model_title = 'VGG16 Model'

    # Summary of the whole model
    #vgg_model = VGG16(
    #    weights='imagenet',
    #    include_top=False,
    #    input_shape=image_params
    #)

    vgg_model = create_vgg_model(image_params)

    head_input = GlobalAveragePooling2D()(vgg_model.output)
    head_output = Dense(units=plant_species_cnt, activation='softmax')(head_input)

    # Visualizing the prediction

    # Get visual model summary
    vgg_model.summary()

    """
    Transfer Learning Model
    
    The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task to solve a specific, smaller task.
    """

    # --- Transfer Learning Model --- #

    # USE THIS
    tl_model_title = 'Transfer Learning Model'

    # Create the transfer learning model
    """
    tl_model = Sequential([
        vgg_model,
        GlobalAveragePooling2D(),  # converts 2D features to 1D vector
        Dense(512, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(plant_species_cnt, activation='softmax')
    ])
    """

    tl_model = create_transfer_learning_model(vgg_model, plant_species_cnt)

    # Unfreeze the top 4 layers
    for layer in tl_model.layers[-4:]:
        layer.trainable = True

    # Recompile with lower learning rate
    tl_model.compile(
        optimizer=Adam(learning_rate=TL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    tl_model.summary()

    start_time = start_timer()
    show_banner(tl_model_title, 'Fitting Training Model')

    tl_model_history = fit_trained_model(
        tl_model,
        x_training_normalized,
        y_training_encoded,
        x_validation_normalized,
        y_validation_encoded,
        x_validation_data,
        training_datagen,
        validation_generator,
        reduce_lr,
        early_stopping
    )

    show_timer(start_time)

    # Plot history
    show_plot_history(tl_model_history, tl_model_title, 'accuracy')
    show_plot_history(tl_model_history, tl_model_title, 'loss')

    """
    Observations:
    
    1. For Transfer Learning Model the acccuracy for validation increased along with the epochs.
    2. The training accuracy didn't increase too much and actually declined. Not good!
    3. The Transfer Learning Model's training loss declined as the epochs increased and the validation losses were more
       consistent.
    4. The model is improving its predictions on the training data by minimizing the error (loss) during back 
       propagation.
    5. The training accuracy is not enough to evaluate the model's performance.
    """

    # Evaluate model
    start_time = start_timer()
    show_banner(tl_model_title, 'Evaluation')

    tl_model_loss, tl_model_accuracy = evalute_model(
        tl_model,
        x_testing_normalized,
        y_testing_encoded
    )

    show_timer(start_time)

    # Model performance classification
    tl_model_training_perf = model_performance_classification(tl_model, x_training_normalized, y_training_encoded)
    print(tl_model_training_perf)

    # Get prediction data for new model
    y_predictor_training, y_predictor_testing, y_testing_normalized = get_model_predictions(
        tl_model,
        x_training_normalized,
        x_testing_normalized,
        y_testing_encoded
    )

    # Plot confusion matrix
    show_plot_confusion_matrix(y_testing_encoded, y_predictor_testing)

    # Display visualization prediction model
    start_time = start_timer()
    prediction_correct, total = show_visualize_prediction(
        tl_model,
        label_encoder,
        x_testing_data,
        x_testing_normalized,
        y_testing_encoded,
        image_height,
        image_width,
        image_channels
    )

    show_timer(start_time)

    pct = (prediction_correct / total) * 100
    show_banner(tl_model_title, f'{prediction_correct} / {total} \nPrediction Accuracy: {pct:.2f}%')

    # Show images of manipulated data prior to training
    show_augmented_image_batch(training_generator, label_encoder)

    show_banner(tl_model_title + ' with '+ vgg_model_title, 'Classification Report')
    print_classification_report(tl_model, x_testing_normalized, y_testing_encoded, plant_species)

    # Conclusions

    print(base_training_perf)
    print(da_model_training_perf)
    print(tl_model_training_perf)

    # ==================================
    #  FINAL RESULT METRICS
    # ==================================
    #
    # Training Accuracy: >= 90%
    # Validation Accuracy: >= 85%
    # Testing Accuracy: >= 85%
    # Testing Loss: <= 50% (want it closes to 0)

    # Build model performance graph.
    models = [base_model_title, da_model_title, tl_model_title]

    # Indicates how well the model is learning the input data.
    training_accuracy = [base_model_history.history['accuracy'][-1], da_model_history.history['accuracy'][-1],
                         tl_model_history.history['accuracy'][-1]]

    # Used during the training process to provide a proxy for generalization.  A gap
    # between this and Training Accuracy is a key sign of overfitting.
    validation_accuracy = [base_model_history.history['val_accuracy'][-1], da_model_history.history['val_accuracy'][-1],
                           tl_model_history.history['val_accuracy'][-1]]

    # MOST IMPORTANT!!!
    # Measures model's performance on completely unseen data that was held out
    # specifically for final evaluation
    testing_accuracy = [base_model_accuracy, da_model_accuracy, tl_model_accuracy]

    # Tests the magnitude of the error the model makes on the unseen test data.
    testing_loss = [base_model_loss, da_model_loss, tl_model_loss]

    # Creating matrix to view final data
    final_data = pd.DataFrame({
        'Models': models,
        'Training Accuracy': training_accuracy,
        'Validation Accuracy': validation_accuracy,
        'Testing Accuracy': testing_accuracy,
        'Testing Loss': testing_loss,
    })

    # --- FINDINGS ----
    print(final_data)

    """
    Actionable Insights and Business Recommendations
    
    * The model correctly identified seedlings.
    * The predictor value for each model was high enough to correctly identify which seedling was which.
    * Augmenting the data with encoders changed the accuracy and the loss dramatically.
    * Transfer Learning helped with the final model in identifying seedlings compared to the augmented one.
    * Training these models at 64x64 sufficed although 128px would probably be better.
    * The size of the filtering helped determine the results.  
    * The base CNN model was the best and efficient model to identify the seedlings.
    * CNN Model accuracy was 95.2%.
    * CNN Model with Augmented Data was 71%.
    * Transfer Learning Model was at 92%.
    * Visualizing the prediction displayed the correct results the majority of the time!
    * The Image Data Manipulator was the biggest factor in determining accuracy.
    * Have better photos that show plants at various stages of growth.
    * Keep backgrounds consistent so that we can focus on the seedling/plants.
    * 12 different seedlings suffice but wouldn't hurt to have a few more.
    * There will be economic benefits with using automation to detect seedlings in the future.
    """

if __name__ == '__main__':
    try:
        run_main_pipeline()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.  Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

# --- End of Program --- #
