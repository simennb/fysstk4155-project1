from functions import *
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creating different run modes in order to easily swap between the different tasks
# Or add an ALL mode to run everything?
run_mode = (input('a/b/c/d/e/f/g/all? ')).lower()
if run_mode not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'all']:
    sys.exit('Please double check input.')


if run_mode == 'a' or run_mode == 'all':
    np.random.seed(4155)
    n_a = 100  # number of points for task a

    # Make meshgrid for the Franke function
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    # Randomly generated meshgrid, TODO: test both maybe?
    x = np.sort(np.random.uniform(0.0, 1.0, n_a))
    y = np.sort(np.random.uniform(0.0, 1.0, n_a))

    x, y = np.meshgrid(x, y)

    z = franke_function(x, y)

    # Adding random noise
    z = z + 0.025 * np.random.randn(20, 20)



if run_mode == 'b' or run_mode == 'all':
    pass
if run_mode == 'c' or run_mode == 'all':
    pass
if run_mode == 'd' or run_mode == 'all':
    pass
if run_mode == 'e' or run_mode == 'all':
    pass
if run_mode == 'f' or run_mode == 'all':
    pass
if run_mode == 'g' or run_mode == 'all':
    pass
