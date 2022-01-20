#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Import cfa removal module
path_file = "path_to_module"
sys.path.insert(1, path_file)
from cfa_removal import cfa_removal

# Datapath
path_in = "/mnt/data_heap/ecg_removal/1_data_preprocessed/"


# Example subject and channel
subj = "VP02"
chan = 40

# Load training data
X_data_rlocked = np.asarray(
    loadmat(os.path.join(path_in, f"{subj}_train_X"))["train_X"]
).astype("float64")
y_data_rlocked = np.asarray(
    loadmat(os.path.join(path_in, f"{subj}_chan{str(chan)}_train_y",))["train_y"]
).astype("float64")

# Load stimlocked data
X_data_stimlocked = np.asarray(
    loadmat(os.path.join(path_in, f"{subj}_predi_X"))["predi_X"]
).astype("float64")
y_data_stimlocked = np.asarray(
    loadmat(os.path.join(path_in, f"{subj}_chan{str(chan)}_predi_y",))["predi_y"]
).astype("float64")


# Remove last 5 trials due to missing values in cardiac phase info
X_data_rlocked = X_data_rlocked[:, :, :-5].copy()
y_data_rlocked = y_data_rlocked[:, :-5].copy()

# Feature data is features x times x trials
# EEG data is times x trials

# Call function
y_data_stimlocked_2d, y_stimlocked_predicted_2d, y_stimlocked_clean_2d = remove_cfa(
    X_data_rlocked,
    y_data_rlocked,
    X_data_stimlocked,
    y_data_stimlocked,
    n_obs_train=1000000,
)


# Plot
plt.plot(y_data_stimlocked_2d.mean(axis=1), label="contaminated", color="r")
plt.plot(y_stimlocked_predicted_2d.mean(axis=1), label="predicted", color="k")
plt.plot(y_stimlocked_clean_2d.mean(axis=1), label="cleaned", color="g")
plt.legend()
plt.show()
