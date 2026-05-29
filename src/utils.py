# utils.py

import time
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src.config import IMAGE_PX_MAX, IMAGE_ROWS, L2_LEARNING_RATE, SECS_IN_MIN, MSEC

def start_timer() -> float:
    return time.time()

def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"

def show_timer(start_time_float: float) -> None:
    print(f"\nRun Time: {get_time(start_time_float)}")

def show_banner(title: str, section: str='') -> None:
    """
    Prints a stylized banner for notebook readability, adjusted to remove
    the trailing hash symbol on the bottom line.
    """
    padding = 2
    title_upper = title.upper()
    # The dashes should match the length of the title plus the two spaces of padding
    dash_count = len(title_upper) + padding

    # Start with a newline for spacing
    print("")
    print('+' + '-' * dash_count + '+')
    print('| ' + title_upper + ' |')
    print('+' + '-' * dash_count + '+')

    # Show section (if provided)
    if section:
        print('' + section)

    print('') # Final newline for spacing

def get_plant_species(df: pd.DataFrame):
    plant_species = sorted(df['Label'].unique().tolist())

    for i, species in enumerate(plant_species):
        print(f'{i}: {species}')

    print(f'\nPlant Species Count: {len(plant_species)}')

    return plant_species

def reduce_lr() -> ReduceLROnPlateau:
    """
    Model Performance Improvement

    Reducing the Learning Rate:

    Hint:
    Use ReduceLRonPlateau() function that will be used to decrease the learning rate by some factor, if the loss is
    not decreasing for some time. This may start decreasing the loss at a smaller learning rate. There is a
    possibility that the loss may still not decrease. This may lead to executing the learning rate reduction again
    in an attempt to achieve a lower loss.
    """

    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=L2_LEARNING_RATE,
        verbose=1
    )

def early_stopping() -> EarlyStopping:
    """
    Early Stopping

    Monitor validation loss and stop training automatically when the loss starts consistently increasing as a sign of
    overfitting.
    """

    return EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
