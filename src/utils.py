# utils.py
import random

import time
#import numpy as np
import pandas as pd
#import tensorflow as tf

from src.config import SECS_IN_MIN, MSEC

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


def get_plant_species(df: pd.DataFrame):
    plant_species = sorted(df['Label'].unique().tolist())

    for i, species in enumerate(plant_species):
        print(f'{i}: {species}')

    print(f'\nPlant Species Count: {len(plant_species)}')

    return plant_species
