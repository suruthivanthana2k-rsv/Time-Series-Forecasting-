import numpy as np
import pandas as pd

def generate_walk_forward_splits(df, initial_train_size, step_size, test_window):
    """
    Yields (train_index_range, val_index_range) tuples for walk-forward.
    initial_train_size: number of samples to use for the first training window
    step_size: how many samples to advance the window each iteration
    test_window: size of each out-of-sample window
    """
    n = len(df)
    start = initial_train_size
    while start + test_window <= n:
        train_idx = (0, start)
        test_idx = (start, start + test_window)
        yield train_idx, test_idx
        start += step_size
