# eda.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History

# Define labeled barplot.
def show_labeled_barplot(data: pd.DataFrame, feature: str, perc: bool=False, n=None) -> None:
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count
    n: displays the top n category levels
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()

    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )
        else:
            label = p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()  # to avoid overlap

# Define confusion matrix
def show_plot_confusion_matrix(y_testing_enc: np.ndarray, y_pred_test: np.ndarray) -> None:

    # Obtaining the categorical values from y_testing_encoded and y_pred
    y_pred_arg = np.argmax(y_pred_test, axis=1)
    y_test_arg = np.argmax(y_testing_enc, axis=1)

    confusion_matrix = tf.math.confusion_matrix(y_test_arg, y_pred_arg)
    f, ax = plt.subplots(figsize=(10, 8))

    plt.title('Confusion Matrix')

    sns.heatmap(
        confusion_matrix,
        annot=True,
        linewidths=0.4,
        fmt='d',
        square=True,
        ax=ax
    )

    plt.show()
    

def show_plot_history(his: History, title: str, column: str) -> None:
    """
    Plots training and validation metrics from a Keras history object
    with enhanced style (ggplot, markers, grid).
    """

    # 🎨 Set a stylish context/style (e.g., 'ggplot' or 'seaborn-v0_8')
    plt.style.use('ggplot')

    # --- Prepare Data and Labels ---
    train_data = his.history[column]
    validation_data = his.history['val_' + column]
    epochs = range(1, len(train_data) + 1)

    metric_title = column.replace('_', ' ').title()
    full_title = f'{title.title()} - {metric_title}'

    # --- Create Plot ---
    plt.figure(figsize=(10, 6)) # Slightly larger for better viewing

    # Plot training data
    plt.plot(
        epochs, train_data,
        label=f'Training {metric_title}',
        color='#1f77b4',       # Blue color
        linewidth=2            # Thicker line
    )

    # Plot validation data with markers
    plt.plot(
        epochs, validation_data,
        label=f'Validation {metric_title}',
        color='#ff7f0e',       # Orange color
        linestyle='--',        # Dashed line
        marker='o',            # Circle markers
        markersize=4           # Small markers
    )

    # --- Add Enhancements ---
    plt.title(full_title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_title, fontsize=12)

    plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)

    # ⚙️ Add a subtle grid
    plt.grid(True, linestyle=':', alpha=0.6)

    # Ensure tight layout and show the plot
    plt.tight_layout()
    plt.show()

def show_plot_histogram(img: np.ndarray, title: str=''):
    """
    Generates an enhanced, appealing histogram of an image's pixel intensities.

    Args:
        img (np.array): The input image array (e.g., grayscale).
        title_text (str): Custom title for the plot.
    """

    # Use a stylish color and add a thin outline (edgecolor)
    plt.hist(
        img.ravel(),
        bins=256,
        color='#4CAF50',  # A pleasant green shade
        edgecolor='#388E3C', # Darker green for the outline
        alpha=0.85, # Slightly higher alpha for more solid color
        linewidth=0.5 # Thin outline
    )

    # Add a clear, prominent Title
    if title:
        title = title.title()

    plt.title(
        title,
        fontsize=16,
        fontweight='bold',
        color='#333333' # Dark gray title color
    )

    # Add X-axis Label
    plt.xlabel(
        'Pixel Value (0 = Black, 255 = White)',
        fontsize=12,
        fontstyle='italic',
        labelpad=10 # Add padding
    )

    # Add Y-axis Label
    plt.ylabel(
        'Frequency (Number of Pixels)',
        fontsize=12,
        fontweight='medium'
    )

    # 5. Add a grid and improve layout
    plt.grid(axis='y', alpha=0.5, linestyle='--') # Add a light, dashed grid on the Y-axis
    plt.xticks(np.arange(0, 257, 32)) # Set clear ticks every 32 units (for better readability)
    plt.xlim([0, 256]) # Ensure the X-axis starts exactly at 0 and ends at 256
    plt.tight_layout() # Adjust plot to prevent labels from being cut off

    # 6. Display the plot
    plt.show()

def show_plant_species_dist(df_labels):

    column_name = 'Label'

    # Assuming df_labels contains the column 'Label' with your classification target.
    plt.figure(figsize=(10, 6))

    # Use the 'y' parameter to create a horizontal bar chart (flipped axes)
    # Use a nice palette for better aesthetics
    ax = sns.countplot(
        y=column_name,
        data=df_labels,
        order=df_labels[column_name].value_counts().index, # Sort bars by count (descending)
        palette='Paired'
    )

    # Add Title and Labels
    ax.set_title('Distribution of Plant Species', fontsize=16, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Plant Species ~ Class', fontsize=12)

    # Add data labels (counts) on the bars
    for container in ax.containers:
        # Set the formatting for the data labels (counts)
        ax.bar_label(container, fmt='%d', padding=5, fontsize=10)

    # Clean up layout and display
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()
