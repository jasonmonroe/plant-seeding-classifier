# main.py

import sys
import warnings

import numpy as np

# TensorFlow and Keras libraries
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Local Source Files
from models.base import BaseModel
from models.cnn_model import CnnModel
from models.data_augm import DataAugmentedModel
from models.transfer_learning import TransferLayerModel
from models.vgg import VggModel

from src.data_handler import DataHandler
from src.eda import show_plot_histogram, show_plant_species_dist, show_labeled_barplot
from src.image_handler import ImageHandler
from src.final_report import FinalReport
from src.utils import get_plant_species, show_banner, start_timer, show_timer


def run_main_pipeline():

    # Fix warnings
    warnings.filterwarnings('ignore')

    print('')
    show_banner('Plant Seedling Classifier')
    print("Number of GPU's Available: ", len(tf.config.list_physical_devices('GPU')))

    # Initialize Data Handler for Plant Seedlings
    plant = DataHandler()
    # @todo - plant.describe_data()
    # @todo - plant.describe_label()
    df_labels = plant.get_labels()

    # Get Plant Species
    plant_species = get_plant_species(df_labels)

    print('Image Information')
    image_width, image_height, image_channels = plant.describe_images()
    images = plant.get_images()

    # Randomly sample pixel data
    image_handle = ImageHandler(images, image_width, image_height, image_channels)
    # @todo - image_handle.show_random_image(images)

    # Plot histogram to check distribution
    # @todo - show_plot_histogram(images)

    """
    Most of the pixels RGB values are between 50-100 with the peak around 75.  This means most of the image data and 
    value is in the middle.
    """

    # Check for NaN values
    print(f'There are {np.isnan(images).sum()} NaN values in the dataset.')

    # Perform Exploratory Data Analysis

    # Plot images like a grid.
    # @todo - image_handle.show_raw_images(df_labels.values.ravel())

    # Label barplot with label
    # @todo - show_labeled_barplot(df_labels, 'Label', perc=True)

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
    # @todo - show_plant_species_dist(df_labels)

    # Data Pre-Processing - Convert the BGR images to RGB images.
    # First, we will display the image as it is imported which means in BGR format.
    # @todo - image_handle.show_random_cv2_image()
    bgr_images = image_handle.create_bgr_images()

    # Now to convert BGR to RGB
    rgb_images = image_handle.convert_to_rgb(bgr_images)

    """
    RGB Image (that has been converted).
    Resize the images
    As the size of the images is large, it may be computationally expensive to train on these larger images; therefore, 
    it is preferable to reduce the image size from 128 to 64.
    
    Note: Your scores will reduce if you lower image size. So it's your choice.
    You can reduce the image in half but for now we will keep same size due to original images having a small filesize.
    """
    reduce_by = 1 # Note: was 2
    resized_img_dims = image_handle.get_resized_img_dims(reduce_by)
    resized_images = image_handle.get_resized_images(rgb_images, resized_img_dims)

    # Random resized image to be shown...
    # @todo - image_handle.show_random_image(resized_images)

    # Create the 3D shape tuple (Height, Width, Channels) for the models
    image_params = resized_img_dims + (image_channels,)
   
    """
    Data Preparation for Modeling
    
    - Before you proceed to build a model, you need to split the data into train, test, and validation to be able to evaluate the model that you build on the train data
    - You'll have to encode categorical features and scale the pixel values.
    - You will build a model using the train data and then check its performance
    
    Split the dataset
    """
    prog_start_time = start_timer()

    dataset = plant.split_data(resized_images)
    dataset['plant_species'] = plant_species

    _encoder = LabelEncoder()
    _encoder.fit(plant_species)
    dataset['_encoder'] = _encoder

    # We use a temporary CnnModel instance to handle the initial 
    # encoding and normalization logic for the entire pipeline.
    cnn = CnnModel(dataset)
    cnn.encode_data()
    cnn.normalize()

    # Retrieve the dataset dictionary now populated with normalized/encoded arrays
    dataset = cnn.get_proc_dataset()

    # --- Build Models --- #

    # Base Model
    base_model = BaseModel(image_params, dataset)
    base_model.run()

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


    """
    Data Augmentation
    
    Note: Data augmentation should not be used in the validation/test data set.
    
    CNN Model with Data Augmentation
    
    The purpose of using a Data Augmented Model (DAM)—or more accurately, applying data augmentation during training—is 
    to artificially increase the size and diversity of your training data without collecting new physical images.
    """

    # Data Augmented CNN Model
    data_augm_model = DataAugmentedModel(image_params, dataset)

    # Augment the data without using validation or test data.  Only training data.
    # The rescale=1./IMAGE_PX_MAX is removed as data is already normalized.
    train_datagen = image_handle.get_train_generator()
    train_generator = image_handle.build_generator(
        train_datagen,
        data_augm_model.x_train_norm,
        data_augm_model.y_train_enc
    )

    data_augm_model.run(train_datagen)

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
    
    Epoch 6: Learning Rate reduced from $1 times 10^{-4 to $5 times 10^{-5 because the validation metrics 
    plateaued or failed to improve.
    
    Epoch 36: Learning Rate reduced again from $5 times 10^{-5 to $2.5 times 10^{-5. This is a common strategy 
    to help the model escape local minima and continue learning, even if very slowly.
    
    Conclusion: 
    This training log confirms the model was stable and effectively prevented overfitting, but the 
    difficulty in learning the highly augmented training data resulted in a lower final performance (the 
    71.58% test accuracy).
    """

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

    # VGG Model
    # Pass the processed dataset so VggModel() can configure its output layers
    vgg_model = VggModel(dataset)
    vgg_model.show_summary()

    # Transfer Learning Model
    tl_model = TransferLayerModel(vgg_model, dataset)
    tl_model.run(train_datagen)

    """
    Observations:
    
    1. For Transfer Learning Model the accuracy for validation increased along with the epochs.
    2. The training accuracy didn't increase too much and actually declined. Not good!
    3. The Transfer Learning Model's training loss declined as the epochs increased and the validation losses were more
       consistent.
    4. The model is improving its predictions on the training data by minimizing the error (loss) during back 
       propagation.
    5. The training accuracy is not enough to evaluate the model's performance.
    """

    image_handle.show_augmented_image_batch(train_generator, _encoder)

    models = [base_model, data_augm_model, tl_model]

    # --- Print Model Performance --- #
    for model in models:
        print(f'{model.title} Training Performance')
        print(model.training_perf)

    #print('Base Model: Training Performances')
    #print(base_model.training_perf)

    #print('Data Augmented Model: Training Performance')
    #print(data_augm_model.training_perf)

    #print('Transfer Layer Model: Training Performance')
    #print(tl_model.training_perf)

    # --- Final Results --- #
    final = FinalReport(models)
    final.output_report()

    print("\n--- End of Program ---")
    show_timer(prog_start_time)

    """
    ### Model Performance Analysis

    1. CNN Base Model (Baseline)
       - Metrics: 59.58% Accuracy | 1.1469 Loss
       - Evaluation: Weak baseline with significant underfitting. 
       - Diagnostic: 3% recall on "Loose Silky-bent" indicates failure on minority classes and bias toward dominant labels.

    2. Data Augmented CNN Model
       - Metrics: 79.58% Accuracy | 0.6053 Loss
       - Evaluation: Structural win. Augmentation forced the model to learn invariant geometric features rather than memorizing pixels.
       - Stability: Tight clustering of Train (80.66%), Val (78.74%), and Test (79.58%) metrics confirms zero overfitting.

    3. VGG16 Transfer Learning Model
       - Metrics: 88.00% Accuracy | 0.4187 Loss
       - Evaluation: Top performer. Pre-trained ImageNet features yielded the lowest loss and highest generalization.
       - Efficiency: Rapidly converged to ~86% validation accuracy early in the training cycle.

    Final Comparison: [Base: 59.58% | Augmented: 79.58% | VGG16: 88.00%]
    """

    """
    Actionable Insights and Business Recommendations
    
    * Model Efficacy: Transfer Learning is the superior strategy for this domain, achieving 88% accuracy by leveraging 
    pre-trained spatial hierarchies that a basic CNN (60%) cannot resolve.
    * Robustness via Augmentation: Implementing geometric transformations (rotation/zoom) was the single most important 
    factor for stability, improving accuracy by 20% and eliminating the overfitting observed in the baseline.
    * Resolution Trade-offs: While 64x64 px enabled rapid iteration and low latency, moving to 128x128 px is recommended 
    for production to better capture the fine-grained leaf serrations necessary for identifying similar species like 
    "Common Wheat" and "Maize."
    * Operational Efficiency: The high generalization across training and testing sets (low variance) suggests the model 
    is ready for integration into automated sorting or monitoring systems.
    
    Future Strategic Improvements:
    * Data Quality: Transition from a static background to a standardized, high-contrast environment to further isolate 
    seedling morphology from noise.
    * Growth Stage Tracking: Expand the dataset to include multi-stage growth photos, as seedling appearance changes 
    significantly between germination and the four-leaf stage.
    * Dataset Balance: The 3% recall on "Loose Silky-bent" in the baseline highlights a need for oversampling minority 
    classes or focused data collection for underrepresented species.
    * Economic Impact: Deploying this automation could significantly reduce labor costs in industrial greenhouses and 
    improve crop yield by enabling early-stage weed detection.
    """


# --- Start Program --- #
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
