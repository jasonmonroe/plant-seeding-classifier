# utils.py
import random

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import SECS_IN_MIN, MSEC, SEED, IMAGE_PX_MAX


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    hours, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
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
    PADDING = 2
    # Ensure title length includes the padding space
    strlen = len(title) + PADDING

    # Start with a newline for spacing
    print("\n")

    # Top line: # ===============
    print('# ', end='')
    print('=' * strlen)

    # Show title: #   Title
    print('#', end='')
    print('  ' + title.upper())

    # Bottom line: # ===============
    print('# ', end='')
    print('=' * strlen)
    # The trailing '#' is now removed here, and the line implicitly ends
    # because the next print statement below will start a new line.

    # Show section (if provided)
    if section:
        print('' + section)

    print('') # Final newline for spacing


    # Clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
def init_cnn_session() -> None:
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

        # Normalize the image(s)
def normalize(img: np.ndarray) -> float:
    return img.astype('float32') / IMAGE_PX_MAX



def get_plant_species(df: pd.DataFrame):
    plant_species = sorted(df['Label'].unique().tolist())

    for i, species in enumerate(plant_species):
        print(f'{i}: {species}')

    return plant_species